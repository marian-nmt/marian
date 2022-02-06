#include <random>

#include "common/file_utils.h"
#include "data/corpus.h"
#include "data/factored_vocab.h"

namespace marian {
namespace data {

typedef std::vector<size_t> WordBatch;
typedef std::vector<float> MaskBatch;
typedef std::pair<WordBatch, MaskBatch> WordMask;
typedef std::vector<WordMask> SentBatch;

void SentenceTupleImpl::setWeights(const std::vector<float>& weights) {
  if(weights.size() != 1) {  // this assumes a single sentence-level weight is always fine
    ABORT_IF(empty(), "Source and target sequences should be added to a tuple before data weights");
    auto numWeights = weights.size();
    auto numTrgWords = back().size();
    // word-level weights may or may not contain a weight for EOS tokens
    if(numWeights != numTrgWords && numWeights != numTrgWords - 1)
      LOG(warn,
          "[warn] "
          "Number of weights ({}) does not match the number of target words ({}) in line #{}",
          numWeights,
          numTrgWords,
          id_);
  }
  weights_ = weights;
}

CorpusIterator::CorpusIterator() : pos_(-1) {}

CorpusIterator::CorpusIterator(CorpusBase* corpus)
    : corpus_(corpus), pos_(0), tup_(corpus_->next()) {}

void CorpusIterator::increment() {
  tup_ = corpus_->next();
  pos_++;
}

bool CorpusIterator::equal(CorpusIterator const& other) const {
  return this->pos_ == other.pos_ || (!this->tup_.valid() && !other.tup_.valid());
}

const SentenceTuple& CorpusIterator::dereference() const {
  return tup_;
}

// These types of corpus constructors are used in in-training validators
// (only?), so do not load additional files for guided alignment or data
// weighting.
CorpusBase::CorpusBase(const std::vector<std::string>& paths,
                       const std::vector<Ptr<Vocab>>& vocabs,
                       Ptr<Options> options,
                       size_t seed)
    : DatasetBase(paths, options), RNGEngine(seed),
      vocabs_(vocabs),
      maxLength_(options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")),
      rightLeft_(options_->get<bool>("right-left")),
      tsv_(options_->get<bool>("tsv", false)),
      tsvNumInputFields_(getNumberOfTSVInputFields(options)) {
  // TODO: support passing only one vocab file if we have fully-tied embeddings
  if(tsv_) {
    ABORT_IF(tsvNumInputFields_ != vocabs_.size(),
             "Number of TSV input fields and vocab files does not agree");
  } else {
    ABORT_IF(paths_.size() != vocabs_.size(),
             "Number of corpus files and vocab files does not agree");
  }

  for(auto path : paths_) {
    UPtr<io::InputFileStream> strm(new io::InputFileStream(path));
    ABORT_IF(strm->empty(), "File '{}' is empty", path);
    files_.emplace_back(std::move(strm));
  }

  initEOS(/*training=*/true);
}

CorpusBase::CorpusBase(Ptr<Options> options, bool translate, size_t seed)
    : DatasetBase(options), RNGEngine(seed),
      maxLength_(options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")),
      rightLeft_(options_->get<bool>("right-left")),
      tsv_(options_->get<bool>("tsv", false)),
      tsvNumInputFields_(getNumberOfTSVInputFields(options)) {
  bool training = !translate;

  if(training)
    paths_ = options_->get<std::vector<std::string>>("train-sets");
  else
    paths_ = options_->get<std::vector<std::string>>("input");

  std::vector<std::string> vocabPaths;
  if(!options_->get<std::vector<std::string>>("vocabs").empty())
    vocabPaths = options_->get<std::vector<std::string>>("vocabs");

  if(training) {
    if(tsv_) {
      ABORT_IF(!vocabPaths.empty() && tsvNumInputFields_ != vocabPaths.size(),
               "Number of TSV input fields and vocab files does not agree");
    } else {
      ABORT_IF(!vocabPaths.empty() && paths_.size() != vocabPaths.size(),
               "Number of corpus files and vocab files does not agree");
    }
  }

  bool useGuidedAlignment = options_->get("guided-alignment", std::string("none")) != "none";
  bool useDataWeighting = options_->hasAndNotEmpty("data-weighting");

  if(training && tsv_) {
    // For TSV input, we expect that guided-alignment or data-weighting provide the index of a TSV
    // field that contains the alignments or weights.
    //
    // Alignments and weights for non TSV input are handled later, after vocab creation.
    if(useGuidedAlignment) {
      try {
        alignFileIdx_ = std::stoul(options_->get<std::string>("guided-alignment"));
      } catch(const std::invalid_argument& /*e*/) {
        ABORT(
            "For TSV input, guided-alignment must provide an index of a field with alignments. "
            "The value '{}' could not be converted to an unsigned integer.",
            options_->get<std::string>("guided-alignment"));
      }
      LOG(info, "[data] Using word alignments from TSV field no. {}", alignFileIdx_);
    }

    if(useDataWeighting) {
      try {
        weightFileIdx_ = std::stoul(options_->get<std::string>("data-weighting"));
      } catch(const std::invalid_argument& /*e*/) {
        ABORT(
            "For TSV input, data-weighting must provide an index of a field with weights. "
            "The value '{}' could not be converted to an unsigned integer.",
            options_->get<std::string>("data-weighting"));
      }
      LOG(info, "[data] Using weights from TSV field no. {}", weightFileIdx_);
    }

    // check for identical or too large indices
    size_t maxIndex = tsvNumInputFields_ + size_t(useGuidedAlignment) + size_t(useDataWeighting) - 1;
    ABORT_IF((useGuidedAlignment && useDataWeighting && alignFileIdx_ == weightFileIdx_)
                 || (useGuidedAlignment && (alignFileIdx_ > maxIndex))
                 || (useDataWeighting && (weightFileIdx_ > maxIndex)),
             "For TSV input, guided-alignment and data-weighting must provide an index <= {} "
             "and be different",
             maxIndex);
  }

  // run this after determining if guided alignment or data weighting is used in TSV input
  initEOS(training);

  // @TODO: check if size_t can be used instead of int
  std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");

  // training or scoring
  if(training) {
    // Marian can create vocabularies automatically if no vocabularies are given or they do not
    // exists under the specified paths.
    //
    // Possible cases:
    //  * -t train1 train2 -v vocab1 vocab2
    //    If vocab1 or vocab2 exists, they are loaded, otherwise separate .yml vocabularies are
    //    created only from train1 or train2 respectively.
    //
    //  * -t train1 train2 -v vocab vocab
    //    If vocab exists, it is loaded, otherwise it is created from concatenated train1 and train2
    //    files.
    //
    //  * -t train1 train2
    //    If no path is given, separate vocabularies train1.yml and train2.yml are created from
    //    train1 and train2 respectively.
    //
    //  * --tsv -t train.tsv -v vocab1 vocab2
    //    If vocab1 or vocab2 exists, it is loaded; otherwise each vocabulary is created from the
    //    appropriate fields in train.tsv.
    //
    //  * --tsv -t train.tsv -v vocab vocab
    //    If vocab exist, it is loaded; otherwise it is created from all fields in train.tsv.
    //
    //  * --tsv -t train.tsv
    //    If no path is given, a train.tsv.yml is created from all fields in train.tsv.
    //
    //  * cat file.tsv | --tsv -t stdin -v vocab1 vocab2
    //    If either vocab1 or vocab2 does not exist, an error is shown that creation of vocabularies
    //    from stdin is not supported.
    //
    //  * cat file.tsv | --tsv -t stdin -v vocab vocab
    //    If vocab does not exist, an error is shown that creation of a vocabulary from stdin is not
    //    supported.
    //
    //  * cat file.tsv | --tsv -t stdin
    //    As above, an error is shown that creation of a vocabulary from stdin is not supported.
    //
    //  There is more cases for multi-encoder models not listed above.
    //
    if(vocabPaths.empty()) {
      size_t numStreams = tsv_ ? tsvNumInputFields_ : paths_.size();

      if(tsv_) {
        // Creating a vocabulary from stdin is not supported
        ABORT_IF(paths_[0] == "stdin" || paths_[0] == "-",
                 "Creating vocabularies automatically from a data stream from STDIN is not "
                 "supported. Create vocabularies first and provide them with --vocabs");

        // Creating a vocab from a TSV input (from STDIN or a file) with alignments or weights is
        // not supported
        ABORT_IF(useGuidedAlignment,
                 "Creating vocabularies automatically from TSV data with alignments is not "
                 "supported. Create vocabularies first and provide them with --vocabs");
        ABORT_IF(useDataWeighting,
                 "Creating vocabularies automatically from TSV data with weights is not "
                 "supported. Create vocabularies first and provide them with --vocabs");
      }

      if(maxVocabs.size() < paths_.size())
        maxVocabs.resize(paths_.size(), 0);

      LOG(info,
          "[data] No vocabulary files given, trying to find or build based on training data.");
      if(!tsv_)
        LOG(info, "[data] Vocabularies will be built separately for each file.");
      else
        LOG(info, "[data] A joint vocabulary will be built from the TSV file.");

      std::vector<int> vocabDims(numStreams, 0);
      std::vector<std::string> vocabPaths1(numStreams);

      // Create vocabs if not provided
      for(size_t i = 0; i < numStreams; ++i) {
        Ptr<Vocab> vocab = New<Vocab>(options_, i);

        const auto& path = paths_[tsv_ ? 0 : i];  // idx 0 because there is always only one TSV file
        std::vector<std::string> trainPaths = {path};
        vocabPaths1[i] = path + ".yml";

        vocabDims[i] = (int) vocab->loadOrCreate("", trainPaths, maxVocabs[i]);
        vocabs_.emplace_back(vocab);
      }
      // TODO: this is not nice as it modifies the option object and needs to expose the changes
      // outside the corpus as models need to know about the vocabulary size; extract the vocab
      // creation functionality from the class.
      options_->set("dim-vocabs", vocabDims, "vocabs", vocabPaths1);

    } else { // Vocabulary paths are given
      size_t numStreams = tsv_ ? tsvNumInputFields_ : paths_.size();

      // Load all vocabs
      size_t numVocs = vocabPaths.size();
      if(maxVocabs.size() < numVocs)
        maxVocabs.resize(numStreams, 0);

      // Helper object for grouping training data based on vocabulary file name
      struct VocabDetails {
        std::set<std::string> paths;  // all paths that are used for training the vocabulary
        std::vector<size_t> streams;  // index of the vocabulary in the --vocab option
        size_t size;                  // the maximum vocabulary size
      };

      // Group training files based on vocabulary path. If the same
      // vocab path corresponds to different training files, this means
      // that a single vocab should combine tokens from all files.
      std::map<std::string, VocabDetails> groupVocab; // vocabPath -> (trainPaths[], vocabSize)
      for(size_t i = 0; i < numVocs; ++i) {
        // Index 0 because there is always only a single TSV input file
        groupVocab[vocabPaths[i]].paths.insert(paths_[tsv_ ? 0 : i]);
        groupVocab[vocabPaths[i]].streams.push_back(i);
        if(groupVocab[vocabPaths[i]].size < maxVocabs[i])
          groupVocab[vocabPaths[i]].size = maxVocabs[i];
      }

      auto vocabDims = options_->get<std::vector<int>>("dim-vocabs");
      vocabDims.resize(numVocs, 0); // make sure there is as many dims as vocab paths

      for(size_t i = 0; i < numVocs; ++i) {
        if(tsv_) {
          bool noVocabGiven = (vocabPaths[i].empty() || !filesystem::exists(vocabPaths[i]));

          // Creating a vocabulary from stdin is not supported
          ABORT_IF(noVocabGiven && (paths_[0] == "stdin" || paths_[0] == "-"),
                   "Creating vocabulary automatically from a data stream from STDIN is not "
                   "supported. Create vocabularies first and provide them with --vocabs");

          // Creating a vocab from a TSV input (from STDIN or a file) with alignments or weights is not supported
          ABORT_IF(noVocabGiven && useGuidedAlignment,
                   "Creating vocabularies automatically from TSV data with alignments is not "
                   "supported. Create vocabularies first and provide them with --vocabs");
          ABORT_IF(noVocabGiven && useDataWeighting,
                   "Creating vocabularies automatically from TSV data with weights is not "
                   "supported. Create vocabularies first and provide them with --vocabs");
        }

        // Get the set of files that corresponds to the vocab. If the next file is the same vocab,
        // it will not be created again, but just correctly loaded.
        auto vocabDetails = groupVocab[vocabPaths[i]];
        std::vector<std::string> groupedPaths(vocabDetails.paths.begin(), vocabDetails.paths.end());
        Ptr<io::TemporaryFile> tsvTempFile;  // temporary handler for cut fields from TSV input

        // For a TSV input, multiple vocabularies with different names mean separate
        // vocabularies for source(s) and target.
        // If a vocabulary does not exist, it will be created in the next step. To be able to create
        // a separate vocabulary, we cut tab-separated field(s) from the TSV file, e.g. all source
        // or target sentences, into a temporary file.
        if(tsv_ && groupVocab.size() > 1 && !filesystem::exists(vocabPaths[i])) {
          ABORT_IF(groupedPaths.size() > 1, "There should not be multiple TSV input files!");

          tsvTempFile.reset(new io::TemporaryFile(options_->get<std::string>("tempdir"), false));
          LOG(info,
              "[data] Cutting field(s) {} from {} into a temporary file {}",
              utils::join(vocabDetails.streams, ", "),
              groupedPaths[0],
              tsvTempFile->getFileName());

          fileutils::cut(groupedPaths[0],  // Index 0 because there is only one TSV file
                         tsvTempFile,
                         vocabDetails.streams,
                         tsvNumInputFields_,
                         " ");  // Notice that tab-separated fields are joined with a whitespace

          groupedPaths.clear();
          groupedPaths.push_back(tsvTempFile->getFileName());
        }

        // Load or create the vocabulary
        Ptr<Vocab> vocab = New<Vocab>(options_, i);
        vocabDims[i] = (int) vocab->loadOrCreate(vocabPaths[i], groupedPaths, vocabDetails.size);
        vocabs_.emplace_back(vocab);

        if(tsvTempFile)
          tsvTempFile.reset();
      }
      // TODO: this is not nice as it modifies the option object and needs to expose the changes
      // outside the corpus as models need to know about the vocabulary size; extract the vocab
      // creation functionality from the class.
      options_->set("dim-vocabs", vocabDims);
    }
  }

  if(translate) {
    ABORT_IF(vocabPaths.empty(), "Translating, but vocabularies are not given!");

    size_t numVocs = vocabPaths.size();
    if(maxVocabs.size() < numVocs)
      maxVocabs.resize(paths_.size(), 0);

    auto vocabDims = options_->get<std::vector<int>>("dim-vocabs");
    vocabDims.resize(numVocs, 0);
    for(size_t i = 0; i + 1 < numVocs; ++i) {
      Ptr<Vocab> vocab = New<Vocab>(options_, i);
      vocabDims[i] = (int) vocab->load(vocabPaths[i], maxVocabs[i]);
      vocabs_.emplace_back(vocab);
    }
    // TODO: As above, this is not nice as it modifies the option object and needs to expose the changes
    // outside the corpus as models need to know about the vocabulary size; extract the vocab
    // creation functionality from the class.
    options_->set("dim-vocabs", vocabDims);
  }

  for(auto path : paths_) {
    if(path == "stdin" || path == "-")
      files_.emplace_back(new std::istream(std::cin.rdbuf()));
    else {
      io::InputFileStream *strm = new io::InputFileStream(path);
      ABORT_IF(strm->empty(), "File '{}' is empty", path);
      files_.emplace_back(strm);
    }
  }

  ABORT_IF(!tsv_ && vocabs_.size() != files_.size(),
           "Number of {} files ({}) and vocab files ({}) does not agree",
           training ? "corpus" : "input",
           files_.size(),
           vocabs_.size());

  // Handle guided alignment and data weighting files. Alignments and weights in TSV input were
  // handled earlier.
  if(training && !tsv_) {
    if(useGuidedAlignment) {
      auto path = options_->get<std::string>("guided-alignment");

      ABORT_IF(!filesystem::exists(path), "Alignment file does not exist");
      LOG(info, "[data] Using word alignments from file {}", path);

      alignFileIdx_ = (int)paths_.size();
      paths_.emplace_back(path);
      io::InputFileStream* strm = new io::InputFileStream(path);
      ABORT_IF(strm->empty(), "File with alignments '{}' is empty", path);
      files_.emplace_back(strm);
    }

    if(useDataWeighting) {
      auto path = options_->get<std::string>("data-weighting");

      ABORT_IF(!filesystem::exists(path), "Weight file does not exist");
      LOG(info, "[data] Using weights from file {}", path);

      weightFileIdx_ = (int)paths_.size();
      paths_.emplace_back(path);
      io::InputFileStream* strm = new io::InputFileStream(path);
      ABORT_IF(strm->empty(), "File with weights '{}' is empty", path);
      files_.emplace_back(strm);
    }
  }
}

void CorpusBase::addWordsToSentenceTuple(const std::string& line,
                                         size_t batchIndex,
                                         SentenceTupleImpl& tup) const {
  // This turns a string in to a sequence of numerical word ids. Depending
  // on the vocabulary type, this can be non-trivial, e.g. when SentencePiece
  // is used.
  Words words = vocabs_[batchIndex]->encode(line, /*addEOS =*/ addEOS_[batchIndex], inference_);

  ABORT_IF(words.empty(), "Empty input sequences are presently untested");

  if(maxLengthCrop_ && words.size() > maxLength_) {
    words.resize(maxLength_);
    if(addEOS_[batchIndex])
      words.back() = vocabs_[batchIndex]->getEosId();
  }

  if(rightLeft_)
    std::reverse(words.begin(), words.end() - 1);

  tup.push_back(words);
}

void CorpusBase::addAlignmentToSentenceTuple(const std::string& line,
                                             SentenceTupleImpl& tup) const {
  ABORT_IF(rightLeft_,
           "Guided alignment and right-left model cannot be used "
           "together at the moment");

  auto align = WordAlignment(line);
  tup.setAlignment(align);
}

void CorpusBase::addWeightsToSentenceTuple(const std::string& line, SentenceTupleImpl& tup) const {
  auto elements = utils::split(line, " ");

  if(!elements.empty()) {
    std::vector<float> weights;
    for(auto& e : elements) {                             // Iterate weights as strings
      if(maxLengthCrop_ && weights.size() >= maxLength_)  // Cut if the input is going to be cut
        break;
      weights.emplace_back(std::stof(e));                 // Add a weight converted into float
    }

    if(rightLeft_)
      std::reverse(weights.begin(), weights.end());

    tup.setWeights(weights);
  }
}

void CorpusBase::addAlignmentsToBatch(Ptr<CorpusBatch> batch,
                                      const std::vector<Sample>& batchVector) {
  int srcWords = (int)batch->front()->batchWidth();
  int trgWords = (int)batch->back()->batchWidth();
  int dimBatch = (int)batch->getSentenceIds().size();

  std::vector<float> aligns(srcWords * dimBatch * trgWords, 0.f);

  for(int b = 0; b < dimBatch; ++b) {

    // If the batch vector is altered within marian by, for example, case augmentation,
    // the guided alignments we received for this tuple cease to be valid.
    // Hence skip setting alignments for that sentence tuple..
    if (!batchVector[b].isAltered()) {
      for(auto p : batchVector[b].getAlignment()) {
        size_t idx = p.srcPos * dimBatch * trgWords + b * trgWords + p.tgtPos;
        aligns[idx] = 1.f;
      }
    }
  }
  batch->setGuidedAlignment(std::move(aligns));
}

void CorpusBase::addWeightsToBatch(Ptr<CorpusBatch> batch,
                                   const std::vector<Sample>& batchVector) {
  int dimBatch = (int)batch->size();
  int trgWords = (int)batch->back()->batchWidth();

  auto sentenceLevel
      = options_->get<std::string>("data-weighting-type") == "sentence";
  size_t size = sentenceLevel ? dimBatch : dimBatch * trgWords;
  std::vector<float> weights(size, 1.f);

  for(int b = 0; b < dimBatch; ++b) {
    if(sentenceLevel) {
      weights[b] = batchVector[b].getWeights().front();
    } else {
      size_t i = 0;
      for(auto& w : batchVector[b].getWeights()) {
        weights[b + i * dimBatch] = w;
        ++i;
      }
    }
  }

  batch->setDataWeights(weights);
}

void CorpusBase::initEOS(bool training = true) {
  // Labels fed into sub-batches that are just class-labels, not sequence labels do not require to
  // add a EOS symbol. Hence decision to add EOS is now based on input stream positions and
  // correspoding input type.

  // Determine the number of streams, i.e. the number of input files (if --train-sets) or fields in
  // a TSV input (if --tsv). Notice that in case of a TSV input, fields that contain alignments and
  // weights are *not* included.
  size_t numStreams = tsv_ ? tsvNumInputFields_ : paths_.size();
  addEOS_.resize(numStreams, true);

  // input-types provides the input type for each input file (if --train-sets) or for each TSV field
  // (if --tsv), for example: sequence, class, alignment.
  auto inputTypes = options_->get<std::vector<std::string>>("input-types", {}); // empty list by default

  // @TODO: think if this should be checked and processed here or in a validation step in config?
  if(!inputTypes.empty()) {
    if(tsv_) {
      // Remove 'alignment' and 'weight' from input types.
      // Note that these input types are not typical input streams with corresponding vocabularies.
      // For a TSV input, they were used only to determine fields that contain alignments or weights
      // and initialize guided-alignment and data-weighting options.
      auto pos = std::find(inputTypes.begin(), inputTypes.end(), "alignment");
      if(pos != inputTypes.end())
        inputTypes.erase(pos);
      pos = std::find(inputTypes.begin(), inputTypes.end(), "weight");
      if(pos != inputTypes.end())
        inputTypes.erase(pos);
    }

    // Make sure there is an input type for each stream
    // and that there is an equal number of input types and streams when training
    ABORT_IF((inputTypes.size() < numStreams) || (training && inputTypes.size() != numStreams),
             "Input types have been specified ({}), you need to specify one per input stream ({})",
             inputTypes.size(), numStreams);
  }

  for(int i = 0; i < numStreams; ++i)
    if(inputTypes.size() > i) {
      if(inputTypes[i] == "class")
        addEOS_[i] = false;
      else if(inputTypes[i] == "sequence")
        addEOS_[i] = true;
      else
        ABORT("Unknown input type {}: {}", i, inputTypes[i]);
    } else {
      // No input type specified, assuming "sequence"
      addEOS_[i] = true;
    }
}

size_t CorpusBase::getNumberOfTSVInputFields(Ptr<Options> options) {
  if(options->get<bool>("tsv", false)) {
    size_t n = options->get<size_t>("tsv-fields", 0);
    if(n > 0 && options->get("guided-alignment", std::string("none")) != "none")
      --n;
    if(n > 0 && options->hasAndNotEmpty("data-weighting"))
      --n;
    return n;
  }
  return 0;
}

#if 0
// experimental: hide inline-fix source tokens from cross attention
std::vector<float> SubBatch::crossMaskWithInlineFixSourceSuppressed() const
{
  const auto& srcVocab = *vocab();

  auto factoredVocab = vocab()->tryAs<FactoredVocab>();
  size_t inlineFixGroupIndex = 0, inlineFixSrc = 0;
  auto hasInlineFixFactors = factoredVocab && factoredVocab->tryGetFactor(FactoredVocab_INLINE_FIX_WHAT_serialized, /*out*/ inlineFixGroupIndex, /*out*/ inlineFixSrc);

  auto fixSrcId = srcVocab[FactoredVocab_FIX_SRC_ID_TAG];
  auto fixTgtId = srcVocab[FactoredVocab_FIX_TGT_ID_TAG];
  auto fixEndId = srcVocab[FactoredVocab_FIX_END_ID_TAG];
  auto unkId = srcVocab.getUnkId();
  auto hasInlineFixTags = fixSrcId != unkId && fixTgtId != unkId && fixEndId != unkId;

  auto m = mask(); // default return value, which we will modify in-place below in case we need to
  if (hasInlineFixFactors || hasInlineFixTags) {
    LOG_ONCE(info, "[data] Suppressing cross-attention into inline-fix source tokens");

    // example: force French translation of name "frank" to always be "franck"
    //  - hasInlineFixFactors: "frank|is franck|it", "frank|is" cannot be cross-attended to
    //  - hasInlineFixTags:    "<IOPEN> frank <IDELIM> franck <ICLOSE>", "frank" and all tags cannot be cross-attended to
    auto dimBatch = batchSize();  // number of sentences in the batch
    auto dimWidth = batchWidth(); // number of words in the longest sentence in the batch
    const auto& d = data();
    size_t numWords = 0;
    for (size_t b = 0; b < dimBatch; b++) {     // loop over batch entries
      bool inside = false;
      for (size_t s = 0; s < dimWidth; s++) {  // loop over source positions
        auto i = locate(/*batchIdx=*/b, /*wordPos=*/s);
        if (!m[i])
          break;
        numWords++;
        // keep track of entering/exiting the inline-fix source tags
        auto w = d[i];
        if (w == fixSrcId)
          inside = true;
        else if (w == fixTgtId)
          inside = false;
        bool wHasSrcIdFactor = hasInlineFixFactors && factoredVocab->getFactor(w, inlineFixGroupIndex) == inlineFixSrc;
        if (inside || w == fixSrcId || w == fixTgtId || w == fixEndId || wHasSrcIdFactor)
          m[i] = 0.0f; // decoder must not look at embedded source, nor the markup tokens
      }
    }
    ABORT_IF(batchWords() != 0/*n/a*/ && numWords != batchWords(), "batchWords() inconsistency??");
  }
  return m;
}
#endif

}  // namespace data
}  // namespace marian

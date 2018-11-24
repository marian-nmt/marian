#include "data/vocab_base.h"

#ifdef USE_SENTENCEPIECE
#include "sentencepiece/src/sentencepiece_processor.h"
#include "sentencepiece/src/sentencepiece_trainer.h"
#endif

#include "common/config.h"
#include "common/options.h"
#include "common/logging.h"
#include "common/filesystem.h"
#include "common/regex.h"

#include <sstream>

namespace marian {

#ifdef USE_SENTENCEPIECE

// Wrapper around https://github.com/google/sentencepiece
class SentencePieceVocab : public VocabBase {
private:
  // Actual SentencePiece processor object
  UPtr<sentencepiece::SentencePieceProcessor> spm_;

  // Sampling factor for subword regularization, disabled when 0
  float alpha_{0};

  // Allowed suffixes for SentencePiece model
  std::vector<std::string> suffixes_ = {".spm"};

  Ptr<Options> options_;
  size_t batchIndex_{0};

  std::mt19937 generator_;
  std::uniform_int_distribution<int> randInt_; // from 0 to INT_MAX

public:
  SentencePieceVocab(Ptr<Options> options, size_t batchIndex)
    : options_(options), batchIndex_(batchIndex), generator_(Config::seed) {

    if(options_->has("sentencepiece-alphas")) {
      auto alphas = options_->get<std::vector<float>>("sentencepiece-alphas");
      if(alphas.size() <= batchIndex)
        alpha_ = 0.f;
      else
        alpha_ = alphas[batchIndex_];

      if(alpha_ > 0)
        LOG(debug,
            "Setting SentencePieceVocab sampling factor to {} for input {}",
            alpha_,
            batchIndex_);
    }

  }

  virtual const std::string& canonicalExtension() const { return suffixes_[0]; }
  virtual const std::vector<std::string>& suffixes() const { return suffixes_; }

  virtual std::string suffix() { return suffixes_[0]; };

  virtual std::string type() const { return "SentencePieceVocab"; }

  virtual Word getEosId() const override { return (Word)spm_->eos_id(); }
  virtual Word getUnkId() const override { return (Word)spm_->unk_id(); }

  void resevoirSampling(std::vector<std::string>& sample, size_t& seenLines,
                        const std::string& trainPath, size_t maxLines, size_t maxBytes) {
    std::unique_ptr<io::InputFileStream> trainStrm(
      trainPath == "stdin" ? new io::InputFileStream(std::cin)
                           : new io::InputFileStream(trainPath)
    );

    std::string line;
    while(getline(*trainStrm, line)) {
      if(line.size() < maxBytes) {
        if(sample.size() < maxLines) {
          sample.push_back(line);
        }
        else {
          size_t i = randInt_(generator_) % (seenLines + 1);
          if(i < maxLines)
            sample[i] = line;
        }
        seenLines++;
      }
    }
  }

  void create(const std::string& vocabPath,
              const std::vector<std::string>& trainPaths,
              size_t maxSize) override {

    size_t defaultMaxSize = 32000;
    size_t maxLines = 10000000;
    size_t maxBytes = 2048;

    if(maxSize == 0) {
      LOG(info, "[data] Vocabulary size is undefined (set with --dim-vocabs ...) - setting to {}", defaultMaxSize);
      maxSize = defaultMaxSize;
    }

    // Iterate over all input files and collect a representative sample via reservoir sampling.
    // The sample will first grow to the desired size and next keep sampling with decreasing
    // probability in the hope to get a uniform sample from the union of all files.
    std::vector<std::string> sample;
    size_t seenLines = 0;
    LOG(info, "[data] Sampling {} lines from {}", maxLines, utils::join(trainPaths, ", "));
    for(const auto& trainPath : trainPaths)
      resevoirSampling(sample, seenLines, trainPath, maxLines, maxBytes);

    // Create temporary file to hold the sample for the SentencePiece trainer
    io::TemporaryFile temp(options_->get<std::string>("tempdir"), false);
    std::string tempFileName = temp.getFileName();
    LOG(info, "[data] Creating temporary file {}", tempFileName);
    {
      io::OutputFileStream out(temp);
      for(const auto& line : sample)
        out << line << std::endl;
    }

    // Compose the SentencePiece training command from filenames and parameters
    std::string sentencePieceOptions = options_->get<std::string>("sentencepiece-options");
    std::stringstream command;
    command
      << " --bos_id=-1 --eos_id=0 --unk_id=1"
      << " --input="        << tempFileName
      << " --model_prefix=" << vocabPath
      << " --vocab_size="   << maxSize
      << " " << sentencePieceOptions;

    // Train the SentencePiece model
    const auto status = sentencepiece::SentencePieceTrainer::Train(command.str());
    ABORT_IF(!status.ok(),
             "SentencePieceVocab error: {}",
             status.ToString());

    LOG(info, "[data] Removing {}", vocabPath + ".vocab");
    ABORT_IF(remove((vocabPath + ".vocab").c_str()) != 0,
             "Could not remove {}",
             vocabPath + ".vocab");

    LOG(info, "[data] Renaming {} to {}", vocabPath + ".model", vocabPath);
    ABORT_IF(rename((vocabPath + ".model").c_str(), vocabPath.c_str()) != 0,
             "Could not rename {} to {}",
             vocabPath + ".model", vocabPath);
  }

  void createFake() {
    ABORT("[data] Fake SentencePieceVocab not supported");
  }

  Word operator[](const std::string& token) const {
    return (Word)spm_->PieceToId(token);
  }

  const std::string& operator[](Word id) const {
    ABORT_IF(id >= size(), "Unknown word id: ", id);
    return spm_->IdToPiece(id);
  }

  Words encode(const std::string& line, bool addEOS, bool inference) const {
    std::vector<int> spmIds;
    if(inference || alpha_ == 0)
      spm_->Encode(line, &spmIds);
    else
      spm_->SampleEncode(line, -1, alpha_, &spmIds);

    Words words(spmIds.begin(), spmIds.end());

    if(addEOS)
      words.push_back(getEosId());
    return words;
  }

  std::string decode(const Words& sentence, bool ignoreEOS) const {
    std::string line;
    // convert vector of Word to vector of int
    std::vector<int> spmSentence(sentence.begin(), sentence.end());
    spm_->Decode(spmSentence, &line);
    return line;
  }

  size_t size() const {
    return spm_->GetPieceSize();
  }

  int load(const std::string& vocabPath, int /*max*/) {
    LOG(info, "[data] Loading SentencePieceVocab from file {}", vocabPath);

    ABORT_IF(!filesystem::exists(vocabPath),
             "SentencePieceVocab file {} does not exits",
             vocabPath);

    spm_.reset(new sentencepiece::SentencePieceProcessor());
    const auto status = spm_->Load(vocabPath);

    ABORT_IF(!status.ok(),
             "SentencePieceVocab error: {}",
             status.ToString());

    return spm_->GetPieceSize();
  }

};
#endif

Ptr<VocabBase> createSentencePieceVocab(const std::string& vocabPath, Ptr<Options> options, size_t batchIndex) {
  bool isSentencePiece = regex::regex_search(vocabPath, regex::regex("\\.(spm)$"));
  if(isSentencePiece) {
#ifdef USE_SENTENCEPIECE
    return New<SentencePieceVocab>(options, batchIndex);
#else
    batchIndex; options;
    ABORT("*.spm suffix in path {} reserved for SentencePiece models, "
          "but support for SentencePiece is not compiled into Marian. "
          "Try to recompile after `cmake .. -DUSE_SENTENCEPIECE=on [...]`",
          vocabPath);
#endif
  }
  // Not a SentencePiece model based on suffix;
  return nullptr;
}

}

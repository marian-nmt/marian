#include "data/vocab_base.h"

#ifdef USE_SENTENCEPIECE
#include "sentencepiece/src/sentencepiece_processor.h"
#include "sentencepiece/src/sentencepiece_trainer.h"
#endif

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

public:
  SentencePieceVocab(Ptr<Options> options, size_t batchIndex)
    : options_(options), batchIndex_(batchIndex) {

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

  void create(const std::string& vocabPath,
              const std::unordered_map<std::string, size_t>& counter,
              size_t /*maxSize*/) override {

    // Create temporary tsv file with vocabulary counts
    io::TemporaryFile temp(options_->get<std::string>("tempdir"), false);
    std::string fileName = temp.getFileName();

    LOG(info, "[data] Creating tsv in temporary file {}", fileName);

    {
      io::OutputFileStream out(temp);
      for(const auto& it : counter)
        out << it.first << "\t" << it.second << std::endl;
    }

    // @TODO: expose parameters
    // @TODO: compute joint vocabs

    std::stringstream command;
    command
      << " --input_format=tsv"
      << " --bos_id=-1 --eos_id=0 --unk_id=1"
      << " --input="        << fileName
      << " --model_prefix=" << vocabPath
      << " --vocab_size=1000";

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

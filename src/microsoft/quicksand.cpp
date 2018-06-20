#include "quicksand.h"
#include "marian.h"

#include "translator/scorers.h"
#include "translator/beam_search.h"

namespace marian {

namespace quicksand {

template <class T>
void set(Ptr<Options> options, const std::string& key, T value) {
    options->set(key, value);
}

template void set(Ptr<Options> options, const std::string& key, size_t);
template void set(Ptr<Options> options, const std::string& key, int);
template void set(Ptr<Options> options, const std::string& key, std::string);
template void set(Ptr<Options> options, const std::string& key, bool);

Ptr<Options> newOptions() {
  return New<Options>();
}

class BeamSearchDecoder : public BeamSearchDecoderBase {
private:
    Ptr<ExpressionGraph> graph_;
    Ptr<Vocab> sourceVocab_;
    Ptr<Vocab> targetVocab_;

    std::vector<Ptr<Scorer>> scorers_;

public:
    BeamSearchDecoder(Ptr<Options> options)
    : BeamSearchDecoderBase(options),
      sourceVocab_(New<Vocab>()),
      targetVocab_(New<Vocab>()) {

        createLoggers();

        sourceVocab_->load(options->get<std::string>("source-vocab"));
        targetVocab_->load(options->get<std::string>("target-vocab"));

        graph_ = New<ExpressionGraph>(true, false);
        graph_->setDevice(DeviceId{0, DeviceType::cpu});
        graph_->reserveWorkspaceMB(500);

        options_->set("inference", true);
        options_->set("ignore-model-config", false);
        options_->set("word-penalty", 0);
        options_->set("normalize", 0);
        options_->set("n-best", false);
        std::string model = options_->get<std::string>("model");

        YAML::Node config;
        Config::GetYamlFromNpz(config, "special:model.yml", model);
        options_->merge(config);

        std::string type = options_->get<std::string>("type");

        auto encdec = models::from_options(options, models::usage::translation);
        scorers_.push_back(New<ScorerWrapper>(encdec, "F0", 1, model));
        for(auto scorer : scorers_) {
          scorer->init(graph_);
        }
    }

    NBest decode(const Sentence& sentence) {
        auto words = (*sourceVocab_)(sentence);
        auto subBatch = New<data::SubBatch>(1, words.size());
        std::copy(words.begin(), words.end(), subBatch->data().begin());
        std::vector<Ptr<data::SubBatch>> subBatches;
        subBatches.push_back(subBatch);
        auto batch = New<data::CorpusBatch>(subBatches);
        batch->setSentenceIds({0});
        batch->debug();

        auto search = New<BeamSearch>(options_, scorers_,
                                      targetVocab_->GetEosId(), targetVocab_->GetUnkId());
        auto histories = search->search(graph_, batch);

        NBest nbest;
        for(auto history : histories) {
          auto bestTranslation = history->Top();
          auto bestTokens = (*targetVocab_)(std::get<0>(bestTranslation));
          auto cost = std::get<1>(bestTranslation)->GetCost();
          nbest.emplace_back(bestTokens, cost);
        }
        return nbest;
    }

};

Ptr<BeamSearchDecoderBase> newDecoder(Ptr<Options> options) {
    return New<BeamSearchDecoder>(options);
}

}
}

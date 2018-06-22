#include "quicksand.h"7
#include "marian.h"

#include "translator/scorers.h"
#include "translator/beam_search.h"

namespace marian {

namespace quicksand {

template <class T>
void set(Ptr<Options> options, const std::string& key, const T& value) {
    options->set(key, value);
}

template void set(Ptr<Options> options, const std::string& key, const size_t&);
template void set(Ptr<Options> options, const std::string& key, const int&);
template void set(Ptr<Options> options, const std::string& key, const std::string&);
template void set(Ptr<Options> options, const std::string& key, const bool&);
template void set(Ptr<Options> options, const std::string& key, const std::vector<std::string>&);

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
        options_->set("word-penalty", 0);
        options_->set("normalize", 0);
        options_->set("n-best", false);

        std::vector<std::string> models = options_->get<std::vector<std::string>>("model");

        for(auto& model : models) {
            Ptr<Options> modelOpts = New<Options>();
            YAML::Node config;
            Config::GetYamlFromNpz(config, "special:model.yml", model);
            modelOpts->merge(options_);
            modelOpts->merge(config);
            auto encdec = models::from_options(modelOpts, models::usage::translation);
            scorers_.push_back(New<ScorerWrapper>(encdec, "F" + std::to_string(scorers_.size()), 1, model));
        }

        for(auto scorer : scorers_) {
          scorer->init(graph_);
        }
    }

    NBest decode(const Sentence& sentence) {
        auto words = (*sourceVocab_)(sentence);

        auto subBatch = New<data::SubBatch>(1, words.size(), sourceVocab_);
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

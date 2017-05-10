#include "marian.h"
#include "translator/translator.h"
#include "translator/beam_search.h"

namespace marian {

std::vector<Ptr<Scorer>>
createScorers(Ptr<Config> options) {
  std::vector<Ptr<Scorer>> scorers;
  
  auto models = options->get<std::vector<std::string>>("models");
  
  int i = 0;
  for(auto model : models) {
    std::string fname = "F" + std::to_string(i++);
    
    auto mOptions = New<Config>(*options);
    mOptions->loadModelParameters(model);
    scorers.push_back(New<ScorerWrapper<MultiHardSoftAtt>>(fname, 1.0f, model, mOptions));
  }
  
  //scorers.push_back(New<WordPenalty>("F2", weights[1], dimVocab));
  //scorers.push_back(New<UnseenWordPenalty>("F3", weights[2], dimVocab, 0));
  
  return scorers;
}

}


int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, true, true);
  
  int dimVocab = options->get<std::vector<int>>("dim-vocabs").back();
      
  auto scorers = createScorers(options);
  auto task = New<TranslateMultiGPU<BeamSearch>>(options, scorers);
  
  task->run();
  
  //WrapModelType<TranslateMultiGPU, BeamSearch>(options)->run();
  
  return 0;

}

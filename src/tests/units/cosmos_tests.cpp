#include "catch.hpp"

#include "microsoft/cosmos.h"
#include "common/definitions.h"
#include "common/filesystem.h"

using namespace marian;

TEST_CASE("microsoft::cosmos::cosine_scorer", "[cosmos]") {
  using namespace cosmos;

  auto logger = spdlog::get("general");
  if(!logger) {
    std::vector<std::string> generalLogs;
    logger = createStderrLogger("general", "[%Y-%m-%d %T] %v", generalLogs, /*quiet=*/true);
  }
  setThrowExceptionOnAbort(true);
  
  auto floatApprox = [](float x, float y) -> bool { 
    return x == Approx(y).margin(0.001f); 
  };

  auto createScorer = [&]() {
    std::string path = "/home/marcinjd/data2/cosmos/embedder/";
    std::string modelPath = path + "2020-07-24.laser.model.npz";
    std::string vocabPath = path + "2020-07-24.laser.vocab.spm";

    CHECK( filesystem::exists(modelPath) );
    CHECK( filesystem::exists(vocabPath) );
    auto scorer = New<MarianCosineScorer>();
    
    CHECK( scorer->load(modelPath, vocabPath) ); 
    
    return scorer;
  };

  auto scorer = createScorer();
  
  SECTION("Compare two identical sentences") {
    std::string input1 = "<CLS> This is a test.";
    std::string input2 = "<CLS> This is a test.";

    auto similarities = scorer->score(input1, input2);

    CHECK( similarities.size() == 1 );
    CHECK( floatApprox(similarities[0], 1.f) );
  }

  SECTION("Compare two different sentences") {
    std::string input1 = "<CLS> This is a test.";
    std::string input2 = "<CLS> This is another test.";

    auto similarities = scorer->score(input1, input2);

    CHECK( similarities.size() == 1 );
    CHECK( floatApprox(similarities[0], 0.94101) );
  }

  SECTION("Compare small batches of sentences") {
    std::string input1 = "<CLS> This is a test.\n<CLS> This is a test.";
    std::string input2 = "<CLS> This is a test.\n<CLS> This is another test.";

    auto similarities = scorer->score(input1, input2);

    CHECK( similarities.size() == 2 );
    CHECK( floatApprox(similarities[0], 1.f) );
    CHECK( floatApprox(similarities[1], 0.94101) );
  }

  SECTION("Throw exception when there is a mismatch in number of sentences (first is shorter)") {  
    std::string input1 = "<CLS> This is a test.\n";
    std::string input2 = "<CLS> This is a test.\n<CLS> This is another test.";

    try {
      marian::setThrowExceptionOnAbort(true);
      auto similarities = scorer->score(input1, input2);
      CHECK( false ); // we shoudn't reach this check, hence a failed test if we do.
    } catch(MarianRuntimeException& e) {
      CHECK( e.what() == std::string("Previous tuple elements are missing.") );
    }
  }

  SECTION("Throw exception when there is a mismatch in number of sentences (second is shorter)") {  
    std::string input1 = "<CLS> This is a test.\n<CLS> This is a test.";
    std::string input2 = "<CLS> This is a test.\n";

    try {
      marian::setThrowExceptionOnAbort(true);
      auto similarities = scorer->score(input1, input2);
      CHECK( false ); // we shoudn't reach this check, hence a failed test if we do.
    } catch(MarianRuntimeException& e) {
      CHECK( e.what() == std::string("There are missing entries in the text tuples.") );
    }
  }
}

TEST_CASE("microsoft::cosmos::embedder", "[cosmos]") {
  using namespace cosmos;

  auto floatApprox = [](float x, float y) -> bool { 
    return x == Approx(y).margin(0.001f); 
  };

  auto createEmbedder = [&]() {
    std::string path = "/home/marcinjd/data2/cosmos/embedder/";
    std::string modelPath = path + "2020-07-24.laser.model.npz";
    std::string vocabPath = path + "2020-07-24.laser.vocab.spm";

    CHECK( filesystem::exists(modelPath) );
    CHECK( filesystem::exists(vocabPath) );
    auto embedder = New<MarianEmbedder>();
    
    CHECK( embedder->load(modelPath, vocabPath) ); 
    
    return embedder;
  };

  auto embedder = createEmbedder();
  
  SECTION("Embed a single sentence") {
    std::string input = "<CLS> This is a test.";
    auto embeddings = embedder->embed(input);

    CHECK( embeddings.size()    ==   1 );
    CHECK( embeddings[0].size() == 512 );

    CHECK( floatApprox(embeddings[0][0], -0.04813f) );
  }

  SECTION("Embed two sentences") {
    std::string input = "<CLS> This is a test.\n<CLS> This is another test.";
    auto embeddings = embedder->embed(input);

    CHECK( embeddings.size()    ==   2 );
    CHECK( embeddings[0].size() == 512 );
    CHECK( embeddings[1].size() == 512 );

    CHECK( floatApprox(embeddings[0][0], -0.04813f) );
    CHECK( floatApprox(embeddings[1][0], -0.04775f) );
  }
}

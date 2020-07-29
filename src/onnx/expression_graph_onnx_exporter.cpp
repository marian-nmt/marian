#ifdef USE_ONNX

#include "onnx/expression_graph_onnx_exporter.h"

#include "models/model_factory.h"
#include "models/encoder_decoder.h"
#include "data/corpus_base.h"
#include "tensors/cpu/fbgemm/expression_graph_packable.h"

#include <memory>

namespace marian {
  // The goal is to export three functions:
  //  - encode_source(): encodes the source
  //                     output: encoder_state
  //  - decode_first(): resets decoder state and performs the first decoding step
  //                    main output: log prob vector for step 0
  //  - decode_next(): performs a subsequent decoding step (called repeatedly)
  //                   main output: log prob vector
  // This is done by generating the tape for encoding followed by the first two decoding steps.
  // As we do this, we remember the Exprs on the tape that are the inputs and the outputs
  // of the three functions.
  // Since not all Marian operations have a 1:1 ONNX counterpart, the tape now get rewritten 
  // such that it only consists of operations that ONNX has.
  // Now we cut out three sub-graphs from the tape. Each sub-graph represents one
  // of the three functions. The sub-graph is delimited by the inputs and outputs we remembered above.
  // Limitations:
  //  - Inner recurrences, e.g. an RNN encoder, are not supported, since we cannot export control flow to ONNX.
  //  - Dynamic objects that depend on the input are not supported.
  //    For example constants whose shape depends on the input length.
  //    That's why we had to change the sinusoidal embeddings from a constant to a computation.
  //  - The input length is represented by a "unique" dimension value (97). This brittle.
  //    That dimension value must not occur naturally in the model.
  //    That dimension must also not be used in dimension calculations.
  //    E.g. the exporter does not recognize if a constant is added to it, or if it gets multiplied.
  void ExpressionGraphONNXExporter::exportToONNX(const std::string& modelToPrefix, Ptr<Options> modelOptions, const std::vector<std::string>& vocabPaths)
  {
    auto graph = shared_from_this();

    // get the model and the vocabularies
    auto model = std::dynamic_pointer_cast<IEncoderDecoder>(models::createModelFromOptions(modelOptions, models::usage::translation));
    std::vector<Ptr<Vocab>> vocabs;
    for (auto vocabPath : vocabPaths) {
      Ptr<Vocab> vocab = New<Vocab>(modelOptions, vocabs.size());
      vocab->load(vocabPath, INT_MAX);
      vocabs.emplace_back(vocab);
    }
    setInference(true);  // note: must also set "inference" parameter on options

    // if we must suppress <unk>, we do that by patching the bias
    const auto trgUnkId = vocabs.back()->getUnkId();
    int unkColId = -1;
    if (trgUnkId != Word::NONE && !modelOptions->get<bool>("allow-unk", false)) { // do we need to suppress unk?
      unkColId = trgUnkId.toWordIndex(); // what's the raw index of unk in the log prob vector?
      // find the bias
      const std::string outputBiasName = "decoder_ff_logit_out_b";
      auto outputBias = graph->get(outputBiasName);
      auto outputBiasVal = outputBias->val();
      std::vector<float> outputBiasVec;
      outputBiasVal->get(outputBiasVec);
      outputBiasVec[unkColId] = -std::numeric_limits<float>::infinity();
      outputBiasVal->set(outputBiasVec);
    }

    // the input length is represented by a value that hopefully is not used elsewhere
    const size_t sentinelDim = 97;  // who uses prime numbers as dimensions anyways!
    size_t numEncoders = vocabs.size() - 1;  // @TODO: test this exporter for >1 encoder

    // some helper functions
    auto extractInputByName = [&](const std::string& name) {
      auto expr = tryFindForwardNodeByName(name);
      ABORT_IF(!expr, "Unexpectedly could not find input node named {}", name);
      expr->set_name("none"); // and nuke the name, as it will be created again in step()
      return std::make_pair(name, expr);
    };
    auto extractEmbeddingInputs = [&](bool forEncoder) {
      // embedding inputs must be found by name, since Marian does not clearly separate batch and Expr version of the batch
      std::vector<std::pair<std::string, Expr>> embeddingInputs;
      for (size_t i = 0; i < numEncoders; i++) {
        // inputs must be found by name, since Marian does not clearly separate batch and Expr version of the batch
        std::string inputName = "data_" + std::to_string(i);
        embeddingInputs.push_back(extractInputByName(inputName));
        if (forEncoder) {
          embeddingInputs.push_back(extractInputByName(inputName + "_mask"));
          embeddingInputs.push_back(extractInputByName(inputName + "_posrange"));
        }
      }
      return embeddingInputs;
    };
    auto extractStates = [&](Ptr<DecoderState> decoderState) {
      std::vector<Expr> states;  // all decoder-state Exprs in a long list
      for (const auto& d : decoderState->getStates()) {
        states.push_back(d.output);
        states.push_back(d.cell);
      }
      return states;
    };

    // run a fake batch through the encoder (this goes into encode_source()) and create decoder start states
    // This adds the operations to the tape.
    std::vector<Ptr<data::SubBatch>> subBatches;
    for (size_t batchIndex = 0; batchIndex < numEncoders; batchIndex++) {
      auto sb = New<data::SubBatch>(1, sentinelDim, vocabs[batchIndex]);
      // set word indices to random values
      std::transform(sb->data().begin(), sb->data().end(), sb->data().begin(),
        [&](Word) -> Word { return vocabs[batchIndex]->randWord(); });
      // mask: no items ask being masked out
      std::fill(sb->mask().begin(), sb->mask().end(), 1.f);
      subBatches.push_back(std::move(sb));
    }
    auto batch = New<data::CorpusBatch>(subBatches);
    auto startState = model->startState(graph, batch);

    // fish out the embedding inputs by name and neutralize the names
    // These constitute the inputs for the graph we are cutting out for encode_source().
    auto encoderEmbeddingInputs = extractEmbeddingInputs(/*forEncoder=*/true);
    std::vector<std::pair<std::string, Expr>> encoderContexts;
    for (const auto& e : startState->getEncoderStates())
      encoderContexts.push_back(std::make_pair("encoder_context_" + std::to_string(encoderContexts.size()), e->getContext()));

    // run it further until the first prediction --> decode_first()
    // This adds more operations to the tape.
    auto decodeFirstState = model->step(graph, startState, /*hypIndices=*/{},
      /*words=*/{}, /*batchIndices=*/{ 0 }, /*beamSize=*/1);
    auto decodeFirstPosRangeInput = extractInputByName("data_" + std::to_string(numEncoders) + "_posrange");

    // run it further until the next prediction --> decode_next()
    // This adds more operations to the tape.
    auto decodeNextState = model->step(graph, decodeFirstState, /*hypIndices=*/{},
      /*words=*/{ vocabs.back()->randWord() }, /*batchIndices=*/{ 0 }, /*beamSize=*/1);
    auto decodeNextEmbeddingInput = extractEmbeddingInputs(/*forEncoder=*/false);
    auto decodeNextPosRangeInput = extractInputByName("data_" + std::to_string(numEncoders) + "_posrange");

    ABORT_IF(encoderContexts.size() != numEncoders, "Unexpected mismatch in number of encoders??");

    // create a descriptor for the three functions, which consists of
    //  - function name
    //  - list of inputs and outputs, as name-Expr pairs
    FunctionDefs functionDefs;

    std::vector<std::pair<std::string, Expr>> inputs;
    std::vector<std::pair<std::string, Expr>> outputs;

    // descriptor for encode_source(data_0, data_0_mask) -> encoder_context_0
    inputs = encoderEmbeddingInputs;
    outputs = encoderContexts;
    functionDefs["encode_source"] = std::make_pair(std::move(inputs), std::move(outputs));

    // descriptor for decode_first(data_1_posrange, encoder_context_0, data_0_mask) -> logits, out_decoder_state_0, out_decoder_state_1, ...
    inputs.emplace_back(decodeFirstPosRangeInput);
    for (size_t i = 0; i < numEncoders; i++) {
      inputs.emplace_back(encoderContexts[i]);
      inputs.emplace_back(encoderEmbeddingInputs[1+2*i]);
    }
    outputs.emplace_back(std::make_pair("first_logits", decodeFirstState->getLogProbs().getLogits()));
    for (const auto& dss : extractStates(decodeFirstState))
      outputs.emplace_back(std::make_pair("first_decoder_state_" + std::to_string(outputs.size()-1), dss));
    functionDefs["decode_first"] = std::make_pair(std::move(inputs), std::move(outputs));

    // descriptor for decode_next(prev_word, data_1_posrange, encoder_context_0, data_0_mask, decoder_state_0, decoder_state_1, ...) -> logits, decoder_state_0, decoder_state_1, ...
    inputs.emplace_back(std::make_pair("prev_word", decodeNextEmbeddingInput[0].second));
    inputs.emplace_back(decodeNextPosRangeInput);
    for (size_t i = 0; i < numEncoders; i++) {
      inputs.emplace_back(encoderContexts[i]);
      inputs.emplace_back(encoderEmbeddingInputs[1 + 2 * i]);
    }
    for (const auto& dss : extractStates(decodeFirstState))
      inputs.emplace_back(std::make_pair("decoder_state_" + std::to_string(inputs.size() - (numEncoders*2 + 2)), dss));
    outputs.emplace_back(std::make_pair("next_logits", decodeNextState->getLogProbs().getLogits()));
    for (const auto& dss : extractStates(decodeNextState))
      outputs.emplace_back(std::make_pair("next_decoder_state_" + std::to_string(outputs.size() - 1), dss));
    functionDefs["decode_next"] = std::make_pair(std::move(inputs), std::move(outputs));

    // now export the sub-graph as given by the function descriptor
    serializeToONNX(modelToPrefix, std::move(functionDefs), sentinelDim);
  }
}

#endif


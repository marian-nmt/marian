#include "common/config_parser.h"
#include "common/definitions.h"

namespace marian {

/**
 * Add all aliases
 *
 * An alias is a command line option name and value pair that sets multiple other non-alias 
 * (standard) command line options. And example would be `--task transformer-big` which -- 
 * as a whole -- is an alias for setting options and hyperparameters that would be reasonable 
 * for training a Google-style Transformer-Big model. Below key-value pairs 
 * ("task", "transformer-base") and ("task", "transformer-big") are different aliases that result
 * in different option sets to be defined.
 * 
 * The alias option has to be first defined using cli.add<T>(). Defining
 * multiple aliases for the same option name but with different values is allowed.
 *
 * As aliases are key-value pairs by default, values are compared as std::string. 
 * If the command line option corresponding to the alias is a vector, the alias 
 * will be triggered if the requested value exists in that vector at least once.
 * By design if an option value that is not defined for that alias option below
 * is used, the CLI parser will abort with 'unknown value for alias' error.
 *
 * @see CLIWrapper::alias()
 *
 * The order of alias definitions *does* matter: options from an alias defined later override
 * options defined in earlier aliases regardless of their order in the command line or config file.
 */
void ConfigParser::addAliases(cli::CLIWrapper& cli) {
  cli.alias("fp16", "true", [&](YAML::Node& config) {
    if(mode_ == cli::mode::training) {
      config["precision"] = std::vector<std::string>({"float16", "float32"}); // inference type, optimization type, save type
      // scaling factor, frequency, multiplier at increase, minium scaling factor
      config["cost-scaling"] = std::vector<std::string>({"256.f", "1000", "2.f", "256.f"});
    } else {
      config["precision"] = std::vector<std::string>({"float16"}); // for inference we do not need the other types
    }
  });

  if(mode_ == cli::mode::training) {
    // for backwards-compatibility with older version, "--no-shuffle" maps to "--shuffle none"
    cli.alias("no-shuffle", "true", [](YAML::Node& config) {
      config["shuffle"] = "none";
    });

    // Options setting the BiDeep architecture proposed in http://www.aclweb.org/anthology/W17-4710
    cli.alias("best-deep", "true", [](YAML::Node& config) {
      config["layer-normalization"] = true;
      config["tied-embeddings"] = true;
      config["enc-type"] = "alternating";
      config["enc-cell-depth"] = 2;
      config["enc-depth"] = 4;
      config["dec-cell-base-depth"] = 4;
      config["dec-cell-high-depth"] = 2;
      config["dec-depth"] = 4;
      config["skip"] = true;

      // Training specific options
      config["learn-rate"] = 0.0003;
      config["cost-type"] = "ce-mean-words";
      config["lr-decay-inv-sqrt"] = 16000;
      config["label-smoothing"] = 0.1;
      config["clip-norm"] = 0;
      config["sync-sgd"] = true;
      config["exponential-smoothing"] = 1e-4;
      config["mini-batch-fit"] = true;
      config["mini-batch"] = 1000;
      config["maxi-batch"] = 1000;
      // config["workspace"] = 6500;
    });

    // Architecture and proposed training settings for a Transformer "base" model introduced in
    // https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    cli.alias("task", "transformer-base", [](YAML::Node& config) {
      // Model options
      config["type"] = "transformer";
      config["enc-depth"] = 6;
      config["dec-depth"] = 6;
      config["dim-emb"] = 512;
      config["tied-embeddings-all"] = true;
      config["transformer-dim-ffn"] = 2048;
      config["transformer-heads"] = 8;
      config["transformer-postprocess"] = "dan";
      config["transformer-preprocess"] = "";
      config["transformer-ffn-activation"] = "relu";
      config["transformer-dropout"] = 0.1;

      // Training specific options
      config["learn-rate"] = 0.0003;
      config["cost-type"] = "ce-mean-words";
      config["lr-warmup"] = 16000;
      config["lr-decay-inv-sqrt"] = 16000;
      config["label-smoothing"] = 0.1;
      config["clip-norm"] = 0;
      config["sync-sgd"] = true;
      config["exponential-smoothing"] = 1e-4;
      config["max-length"] = 100;
      config["mini-batch-fit"] = true;
      config["mini-batch"] = 1000;
      config["maxi-batch"] = 1000;
      config["workspace"] = 9500;
      config["optimizer-params"] = std::vector<float>({0.9f, 0.98f, 1e-09f});

      // Validation specific options
      config["beam-size"] = 8;
      config["valid-mini-batch"] = 16;
      config["normalize"] = 1.0;
    });

    // Architecture and proposed training settings for a Transformer "big" model introduced in
    // https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    cli.alias("task", "transformer-big", [](YAML::Node& config) {
      // Model options
      config["type"] = "transformer";
      config["enc-depth"] = 6;
      config["dec-depth"] = 6;
      config["dim-emb"] = 1024;
      config["tied-embeddings-all"] = true;
      config["transformer-dim-ffn"] = 4096;
      config["transformer-heads"] = 16;
      config["transformer-postprocess"] = "dan";
      config["transformer-preprocess"] = "";
      config["transformer-ffn-activation"] = "relu";
      config["transformer-dropout"] = 0.1;

      // Training specific options
      config["learn-rate"] = 0.0002;
      config["cost-type"] = "ce-mean-words";
      config["lr-warmup"] = 8000;
      config["lr-decay-inv-sqrt"] = 8000;
      config["label-smoothing"] = 0.1;
      config["clip-norm"] = 0;
      config["sync-sgd"] = true;
      config["exponential-smoothing"] = 1e-4;
      config["max-length"] = 100;
      config["mini-batch-fit"] = true;
      config["mini-batch"] = 1000;
      config["maxi-batch"] = 1000;
      config["workspace"] = 13000;
      config["optimizer-params"] = std::vector<float>({0.9f, 0.998f, 1e-09f});

      // Validation specific options
      config["beam-size"] = 8;
      config["valid-mini-batch"] = 8;
      config["normalize"] = 1.0;
    });

    // Transformer base variant with "prenorm" (i.e. the layer normalization is performed as the first block-wise
    // preprocessing step). This also requires to normalize the final output of a transformer stack to avoid the 
    // activations to blow up. This blow up is particularly nasty with mixed precision training.
    // See implementation and comments in tensor2tensor: 
    // * https://github.com/tensorflow/tensor2tensor/blob/95d021477272c10af15cd62f25b595ad16ad514e/tensor2tensor/models/transformer.py#L1845
    // * https://github.com/tensorflow/tensor2tensor/commit/f5c9b17e617ea9179b7d84d36b1e8162cb369f25#diff-4e58a582cf11ca649e76b4362d69e405R78
    cli.alias("task", "transformer-base-prenorm", [](YAML::Node& config) {
      // Model options
      config["type"] = "transformer";
      config["enc-depth"] = 6;
      config["dec-depth"] = 6;
      config["dim-emb"] = 512;
      config["tied-embeddings-all"] = true;
      config["transformer-dim-ffn"] = 2048;
      config["transformer-heads"] = 8;
      config["transformer-postprocess"] = "da";     // change from transformer-base is "dan" -> "da"
      config["transformer-preprocess"] = "n";       // change from transformer-base is "" -> "n"
      config["transformer-postprocess-top"] = "n";  // change from transformer-base is "" -> "n"
      config["transformer-ffn-activation"] = "relu";
      config["transformer-dropout"] = 0.1;

      // Training specific options
      config["learn-rate"] = 0.0003;
      config["cost-type"] = "ce-mean-words";
      config["lr-warmup"] = 16000;
      config["lr-decay-inv-sqrt"] = 16000;
      config["label-smoothing"] = 0.1;
      config["clip-norm"] = 0;
      config["sync-sgd"] = true;
      config["exponential-smoothing"] = 1e-4;
      config["max-length"] = 100;
      config["mini-batch-fit"] = true;
      config["mini-batch"] = 1000;
      config["maxi-batch"] = 1000;
      config["workspace"] = 9500;
      config["optimizer-params"] = std::vector<float>({0.9f, 0.98f, 1e-09f});

      // Validation specific options
      config["beam-size"] = 8;
      config["valid-mini-batch"] = 16;
      config["normalize"] = 1.0;
    });

    // Transformer big variant with "prenorm". Same changes as above.
    cli.alias("task", "transformer-big-prenorm", [](YAML::Node& config) {
      // Model options
      config["type"] = "transformer";
      config["enc-depth"] = 6;
      config["dec-depth"] = 6;
      config["dim-emb"] = 1024;
      config["tied-embeddings-all"] = true;
      config["transformer-dim-ffn"] = 4096;
      config["transformer-heads"] = 16;
      config["transformer-postprocess"] = "da";     // change from transformer-big is "dan" -> "da"
      config["transformer-preprocess"] = "n";       // change from transformer-big is "" -> "n"
      config["transformer-postprocess-top"] = "n";  // change from transformer-big is "" -> "n"
      config["transformer-ffn-activation"] = "relu";
      config["transformer-dropout"] = 0.1;

      // Training specific options
      config["learn-rate"] = 0.0002;
      config["cost-type"] = "ce-mean-words";
      config["lr-warmup"] = 8000;
      config["lr-decay-inv-sqrt"] = 8000;
      config["label-smoothing"] = 0.1;
      config["clip-norm"] = 0;
      config["sync-sgd"] = true;
      config["exponential-smoothing"] = 1e-4;
      config["max-length"] = 100;
      config["mini-batch-fit"] = true;
      config["mini-batch"] = 1000;
      config["maxi-batch"] = 1000;
      config["workspace"] = 13000;
      config["optimizer-params"] = std::vector<float>({0.9f, 0.998f, 1e-09f});

      // Validation specific options
      config["beam-size"] = 8;
      config["valid-mini-batch"] = 8;
      config["normalize"] = 1.0;
    });
  }
}

}  // namespace marian

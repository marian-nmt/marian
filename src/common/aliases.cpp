#include "common/config_parser.h"
#include "common/definitions.h"

namespace marian {

/**
 * Add all aliases
 *
 * An alias is a shortcut option for a predefined set of options. It is triggered if the option has
 * the requested value. The alias option has to be first defined using cli.add<T>(). Defining
 * multiple aliases for the same option name but with different value is allowed.
 *
 * Values are compared as std::string. If the alias option is a vector, the alias will be triggered
 * if the requested value exists in that vector at least once.
 *
 * @see CLIWrapper::alias()
 *
 * The order of alias definitions *does* matter: options from later aliases override earlier
 * regardless of its order in the command line or config file.
 */
void ConfigParser::addAliases(cli::CLIWrapper& cli) {
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
  });

  // Architecture and proposed training settings for a Transformer BASE model introduced in
  // https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
  cli.alias("task", "transformer", [](YAML::Node& config) {
    config["type"] = "transformer";
    config["enc-depth"] = 6;
    config["dec-depth"] = 6;
    config["transformer-heads"] = 8;
    config["learn-rate"] = 0.0003;
    config["cost-type"] = "ce-mean-words";
    config["lr-warmup"] = 16000;
    config["lr-decay-inv-sqrt"] = 16000;
    config["transformer-dropout"] = 0.1;
    config["label-smoothing"] = 0.1;
    config["clip-norm"] = 5;
  });

  // Architecture and proposed training settings for a Transformer BIG model introduced in
  // https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
  cli.alias("task", "transformer-big", [](YAML::Node& config) {
    config["type"] = "transformer";
    config["enc-depth"] = 6;
    config["dec-depth"] = 6;
    config["dim-emb"] = 1024;
    config["transformer-dim-ffn"] = 4096;
    config["transformer-heads"] = 16;
    config["transformer-postprocess"] = "dan";
    config["transformer-preprocess"] = "d";
    config["transformer-ffn-activation"] = "relu";
    config["learn-rate"] = 0.0002;
    config["cost-type"] = "ce-mean-words";
    config["lr-warmup"] = 8000;
    config["lr-decay-inv-sqrt"] = 8000;
    config["transformer-dropout"] = 0.1;
    config["transformer-dropout-attention"] = 0.1;
    config["transformer-dropout-ffn"] = 0.1;
    config["label-smoothing"] = 0.1;
    config["clip-norm"] = 5;
  });
}

}  // namespace marian

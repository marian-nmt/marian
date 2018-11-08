#pragma once

#include <string>
#include "marian.h"

namespace marian {
namespace models {

enum struct usage { raw, training, scoring, translation };
}
}  // namespace marian

YAML_REGISTER_TYPE(marian::models::usage, int)

namespace marian {
namespace models {

class ModelBase {
public:
  virtual void load(Ptr<ExpressionGraph>,
                    const std::string&,
                    bool markReloaded = true) const 
      = 0; // changes graph but not model, therefore declared const
  
  virtual void save(Ptr<const ExpressionGraph>,
                    const std::string&,
                    bool saveTranslatorConfig = false) const
      = 0; // doesn't change the model, therefore const

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true)
      = 0;

  virtual void clear(Ptr<ExpressionGraph> graph) = 0;
};

}  // namespace models
}  // namespace marian

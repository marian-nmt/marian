#include "graph/expression_graph.h"

namespace marian {
  // export of Marian models to ONNX
  class ExpressionGraphONNXExporter : public ExpressionGraph {
#ifdef USE_ONNX
    public:
    // export a seq2seq model to a set of ONNX files
    void exportToONNX(const std::string& modelToPrefix, Ptr<Options> modelOptions, const std::vector<std::string>& vocabPaths);

  private:
    // [name] -> (vector(name, Expr), vector(name, Expr))
    typedef std::map<std::string, std::pair<std::vector<std::pair<std::string, Expr>>, std::vector<std::pair<std::string, Expr>> >> FunctionDefs;

    // serialize the current nodesForward_ to an ONNX file. This operation is destructive.
    void serializeToONNX(const std::string& filename, FunctionDefs&& functionDefs, size_t sentinelDim);

    // find a node on the current forward tape
    Expr tryFindForwardNodeByName(const std::string& nodeName) const;

    // helper to transform nodesForward_ to only use the subset of operations supported by ONNX
    void expandMacroOpsForONNX(std::map<std::string, std::pair<std::vector<std::pair<std::string, Expr>>, std::vector<std::pair<std::string, Expr>> >>& functionDefs);

    // helper to build nodesForward_ from root nodes
    void rebuildNodesForward(const struct InputsMap& inputsMap,
                             const std::vector<std::pair<std::string, Expr>>& outputDefs);
#endif // USE_ONNX
  };
}

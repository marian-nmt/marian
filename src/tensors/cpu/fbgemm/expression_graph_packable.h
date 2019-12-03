#pragma once

#include "graph/expression_graph.h"
#include "packed_gemm.h"

namespace marian {

// When FBGEMM based packed GEMM is used, some weight matrices need to be packed offline.
// The decision which weights can be packed or not should be done walking through the graph.
// This requires some more changes, but we temporarily do this just by name ("_W") of the weights.
// And, this introduces a low level packed_gemm.h apis interact with high level graph class.
// So, we make a subclass of ExpressionGraph and put those immature codes in this class.
// We will improve this in the near future. 
class ExpressionGraphPackable : public ExpressionGraph {
public:
  ExpressionGraphPackable()
    : ExpressionGraph( /* inference =  */ true) {} // Packable expression graph only supports inference

  virtual ~ExpressionGraphPackable() {}

  // Convert model weights into packed format and save to IO items.
  // @TODO: review this
  void packAndSave(const std::string& name, const std::string& meta, std::string& saveGemmType, Type saveElementType = Type::float32) {
    std::vector<io::Item> ioItems;

    // sorted by name in std::map
    for (auto p : params()->getMap()) {
      std::string pName = p.first;

      if (!namespace_.empty()) {
        if (pName.substr(0, namespace_.size() + 2) == namespace_ + "::")
          pName = pName.substr(namespace_.size() + 2);
      }

      Tensor val = p.second->val();

#if USE_FBGEMM
      // save as packed format
      // @TODO Hardcoded to find packable weights - all the weights used for affine op (fp16), all the weights used for affine op and dot op (int8)
      if (saveGemmType == "int8packed" && (pName.find("_W") == pName.length() - 3 || pName.find("_W") == pName.length() - 2))
      {
        using namespace marian::cpu::variant;

        // packing information - size
        int nrow;
        int ncol;
        uint64_t packsize;

        fbgemmPacked8PackInfo(val->shape(),
          pName.find("Wemb") != std::string::npos,
          nrow,
          ncol,
          packsize);

        auto allocator = New<TensorAllocator>(getBackend());

        // buffer tensor to save packed matrix
        Tensor packedTensor;
        allocator->allocate(packedTensor, { 1, (int32_t)packsize }, Type::uint8);

        //Pack B matrix into int8
        fbgemmPacked8Pack(packedTensor,
          val->data(),
          pName.find("Wemb") != std::string::npos,
          nrow,
          ncol,
          packsize);
        io::Item item;
        item.name = pName;
        item.shape = val->shape();
        item.type = Type::packed8;

        // Use the actual memory as this will be aligned and padded.
        // When memory mapping this is required. Shape keeps track of
        // tensor size. Saving to *.npz will cut to size.
        auto mem = packedTensor->memory();
        item.bytes.resize(mem->size());
        copy(backend_, mem->data<char>(), mem->data<char>() + mem->size(), item.bytes.data());

        ioItems.emplace_back(std::move(item));

      } else if (saveGemmType == "fp16packed" && pName.find("_W") == pName.length() - 3)
      {
        using namespace marian::cpu::variant;

        // packing information
        int nrow, ncol, kernel_ncol_blocks, brow, bcol, last_brow, nbrow, nbcol;
        uint64_t packsize;

        fbgemmPacked16PackInfo(val->shape(),
          false,
          nrow,
          ncol,
          kernel_ncol_blocks,
          brow,
          bcol,
          last_brow,
          nbrow,
          nbcol,
          packsize);

        auto allocator = New<TensorAllocator>(getBackend());

        Tensor packedTensor;
        allocator->allocate(packedTensor, { 1, (int32_t)packsize }, Type::uint8);

        // fbgemmPacked16Pack
        fbgemmPacked16Pack(packedTensor,
          val->data(),
          false,
          nrow,
          ncol,
          kernel_ncol_blocks,
          brow,
          bcol,
          last_brow,
          nbrow,
          nbcol,
          packsize);
        io::Item item;
        item.name = pName;
        item.shape = val->shape();
        item.type = Type::packed16;

        // Use the actual memory as this will be aligned and padded.
        // When memory mapping this is required. Shape keeps track of
        // tensor size. Saving to *.npz will cut to size.
        auto mem = packedTensor->memory();
        item.bytes.resize(mem->size());
        copy(backend_, mem->data<char>(), mem->data<char>() + mem->size(), item.bytes.data());

        ioItems.emplace_back(std::move(item));
      } else
#endif  // USE_FBGEMM
      {
        io::Item item;
        val->get(item, pName);
        item.convert(saveElementType);
        ioItems.emplace_back(std::move(item));
      }
    }

    if (!meta.empty())
      io::addMetaToItems(meta, "special:model.yml", ioItems);
    io::saveItems(name, ioItems);
  }
};

}  // namespace marian
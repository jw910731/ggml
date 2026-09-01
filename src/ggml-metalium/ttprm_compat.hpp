#pragma once

// Compatibility shim for building ttPseudoRowMajor against this tree's tt-metal.
//
// ttprm targets an older TTNN header layout than the tt-metal this backend links
// (v0.78.0-dev). Rather than fork the submodule, re-establish the spellings it
// expects. This header is force-included into ttprm's translation unit (see
// CMakeLists.txt) and included directly by ttprm_bridge.cpp.
//
// Each alias below exists because ttprm names a symbol that tt-metal has since
// moved; if a future tt-metal restores the original spelling the alias becomes a
// harmless no-op typedef, and if ttprm is updated upstream this file can go away.

#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/tensor/types.hpp>

namespace tt::tt_metal {
// StorageType moved from tt::tt_metal into ttnn.
using StorageType = ::ttnn::StorageType;
// So did create_device_tensor.
using ::ttnn::create_device_tensor;
}  // namespace tt::tt_metal

namespace ttnn {
// The ttnn::TensorSpec re-export was dropped; it lives in tt::tt_metal now.
using TensorSpec = ::tt::tt_metal::TensorSpec;
}  // namespace ttnn

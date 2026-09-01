#pragma once

// Bridge between the ggml-metalium backend and ttPseudoRowMajor (ttprm).
//
// TTNN has no view: this backend materializes every VIEW/RESHAPE node into a
// fresh DRAM tensor (realize_ggml_view_impl) and every consumer pays for it.
// ttprm exists to avoid exactly that -- it folds a view chain on the HOST into a
// single affine map
//
//     k(oR, oC) = base + oR*S_row + oC          (inner/lane stride 1)
//
// and hands that map to a fused device kernel, so the view is walked while the
// real work happens instead of being copied out first.
//
// This file translates a ggml view chain into a ttprm::View and exposes the two
// entry points the backend uses:
//
//   ggml_ttprm::realize_view()  one gather instead of reshape/slice/reshape
//   ggml_ttprm::bin_op()        realize(a) + realize(b) + ttnn::add  ->  ONE
//                               program with zero intermediate tensors
//   ggml_ttprm::norm()          reduce straight out of the view
//
// All of them are best-effort. Every entry point returns nullptr when the view falls
// outside what can be done safely, and the caller runs the existing path unchanged.
// Three separate things narrow the gate:
//
//   * ttprm's own accept gate -- bf16 + TILE + device-resident, lane stride 1,
//     face-aligned base and strides, affine row map -- reported through a
//     non-throwing Result.
//   * The interior-padding trap: ttprm::View silently anchors on the PADDED grid when
//     a middle axis is padded, which would move logical row r away from row r. The
//     bridge reproduces that decision (anchor_grid) and refuses the shapes where the
//     flat element index would stop matching ggml's.
//   * Whether the result can carry the node's shape without a physical re-tile.
//
// The Blackhole 64-byte NoC phase rule used to be gated here too -- ttprm moves 32-byte
// face-rows, and an L1<->DRAM transfer is only legal when both addresses agree modulo
// NOC_DRAM_READ_ALIGNMENT_BYTES (32 on Wormhole, 64 on Blackhole). That is now fixed
// inside ttprm's gather itself (kernel/tensor_view.hpp bounces the offending face-rows),
// so no gate is needed here.
//
// Set GGML_METALIUM_TTPRM_STATS=1 to print the accept/reject histogram at exit; the
// reject reasons are the cheapest way to see whether widening the gate is worthwhile.
//
// Compiled only when GGML_METALIUM_TTPRM is defined; see CMakeLists.txt.

#include <memory>

#include "ggml.h"
#include "ttnn/tensor/tensor.hpp"

namespace ggml_ttprm {

// Runtime kill switch. True unless GGML_METALIUM_TTPRM is set to 0/false/no/off.
bool enabled();

// Resolve a ggml view chain and relayout it in a single ttprm gather.
// Returns nullptr when the chain is not representable -- the caller must then run
// the existing realize path.
std::shared_ptr<ttnn::Tensor> realize_view(const ggml_tensor* node);

// Fused elementwise binary. `dst` must be an ADD/SUB/MUL node; both operands are
// consumed as views, so a viewed operand is never materialized.
// Returns nullptr when the operands fall outside the gate.
std::shared_ptr<ttnn::Tensor> bin_op(const ggml_tensor* dst, ggml_op op);

// Fused RMS/layer norm reducing straight out of a viewed source, instead of
// realizing the view and then normalizing it.
// Returns nullptr when the source falls outside the gate.
std::shared_ptr<ttnn::Tensor> norm(const ggml_tensor* dst, bool rms, float eps);

}  // namespace ggml_ttprm

// Implemented by ggml-metalium.cpp: the ttnn::Tensor already materialized for
// `node`, or nullptr if the node has none (it is a view, or not computed yet).
// The bridge cannot see ggml_tensor_extra_metalium, which is backend-private.
const ttnn::Tensor* ggml_metalium_materialized_tensor(const ggml_tensor* node);

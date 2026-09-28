#pragma once

// Host-only layout algebra for the Metalium backend: which row-local data movement a ggml view
// chain is, planned before any device op runs. Nothing here touches TTNN, so the planner can be
// tested against the CPU backend on its own.
//
// A TILE tensor of TT shape [..., R, W] is seen as `blocks` blocks of R rows of W lanes. Its
// row-major element order is the ggml order of the tensor it stores, natural or folded.

#include "ggml.h"

#include <array>
#include <cstdint>
#include <vector>

struct ml_grid {
    int64_t blocks = 1;
    int64_t R      = 1;
    int64_t W      = 1;
};

// dims in TT order, outermost first
ml_grid ml_grid_of(const int64_t * tt_shape, int rank);

// The TT shape of ne with its g lowest dims merged into the lane dim (g = 1 is the natural shape)
std::array<int64_t, 4> ml_coarsen(const int64_t * ne, int g);

// Whether ttnn::reshape of a TILE tensor between two TT shapes of equal volume only relabels it
bool ml_reshape_is_view(const std::array<int64_t, 4> & from, const std::array<int64_t, 4> & to);

// Element i of a node reads base element offset + sum_d (i_d % ne_src[d]) * st[d] of the base's flat order
struct ml_access {
    const ggml_tensor * base = nullptr;
    int64_t             ne[GGML_MAX_DIMS];
    int64_t             ne_src[GGML_MAX_DIMS];  // ne, or what a REPEAT tiles up to ne
    int64_t             st[GGML_MAX_DIMS];
    int64_t             offset = 0;
};

// Walks t's src[0] chain through VIEW/RESHAPE/PERMUTE/TRANSPOSE to the first other node, the base,
// summing the VIEW nodes' own offsets. Returns nullptr, or why the chain cannot be resolved.
const char * ml_resolve_chain(const ggml_tensor * t, ml_access & acc);

// Access of each element of a REPEAT node over the base of its source's chain
const char * ml_resolve_repeat(const ggml_tensor * dst, ml_access & acc);

enum class ml_lane_op {
    identity,  // lanes[c] == c over the whole base row
    window,    // lanes[c] == lanes[0] + c with lanes[0] tile-aligned: a last-dim slice
    gather,    // anything else
};

struct ml_plane {
    int64_t              blk0 = 0, nblk = 0;  // base blocks read
    int64_t              row0 = 0, nrow = 0;  // rows read in each of those blocks
    ml_lane_op           op = ml_lane_op::gather;
    std::vector<int32_t> lanes;               // output lane c is base lane lanes[c]; -1 is zero
    int                  dup_of = -1;         // the same result as an earlier plane
};

// The result is the planes' [nblk, nrow, W_out] pieces concatenated in order, which is out_shape's
// row-major order: each output row reads lanes of one base row, rows keep their base order.
struct ml_plan {
    ml_grid                in;
    std::array<int64_t, 4> out_shape{};
    int64_t                W_out = 0;
    std::vector<ml_plane>  planes;

    bool identity() const;  // the result is the base itself, relabelled
};

// Plans the node an access describes as the TT shape ml_coarsen(acc.ne, g_out) read out of a base
// laid out as `in`. Every check happens here. Returns nullptr, or why the node is not row-local.
const char * ml_plan_row_local(const ml_access & acc, int g_out, const ml_grid & in, ml_plan & plan);

// For each (plane, 32-lane output tile): a PAGE run copies one whole tile-aligned base tile
struct ml_run {
    int     plane;
    int64_t out_tile;
    bool    page;
    int64_t src_tile;  // for a PAGE run
};
std::vector<ml_run> ml_classify_runs(const ml_plan & plan);

// CPU oracle: the plan applied to the base's elements in flat order
std::vector<float> ml_apply_plan_host(const ml_plan & plan, const std::vector<float> & base);

// Whether a binary op can take src1 folded like dst (g lowest dims as lanes): src1 may broadcast
// row and outer dims, or all of its lane dims at once, but never part of a lane group.
const char * ml_bin_broadcast(const int64_t * dst_ne, const int64_t * src1_ne, int g);

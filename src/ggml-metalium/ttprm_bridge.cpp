// ggml view chain -> ttprm::View translation. See ttprm_bridge.hpp.

#include "ttprm_bridge.hpp"

#include <atomic>
#include <cstdlib>
#include <numeric>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "ttprm_compat.hpp"

#include "ttprm.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ggml_ttprm {
namespace {

// ---------------------------------------------------------------------------
// Flags and accept/reject accounting.
//
// The reject histogram is the cheap experiment: run a real graph with
// GGML_METALIUM_TTPRM_STATS=1 and the reasons say whether widening the gate is
// worth any further work, before building anything on top of it.
// ---------------------------------------------------------------------------

bool parse_env_default(const char* name, bool dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr) return dflt;
    std::string s(v);
    for (char& c : s) c = (char)::tolower((unsigned char)c);
    return !(s == "0" || s == "false" || s == "no" || s == "off");
}

struct stats_t {
    std::mutex mu;
    std::map<std::string, size_t> counts;

    void bump(const std::string& key) {
        std::lock_guard<std::mutex> lock(mu);
        counts[key]++;
    }

    ~stats_t() {
        if (!parse_env_default("GGML_METALIUM_TTPRM_STATS", false)) return;
        if (counts.empty()) return;
        fmt::println(stderr, "\n[ttprm] accept/reject histogram:");
        for (const auto& [k, v] : counts) fmt::println(stderr, "  {:>8}  {}", v, k);
    }
};

stats_t g_stats;

// ttprm's device kernels are JIT-compiled from TTPRM_ROOT/kernel/*.cpp at first use.
// A kernel that no longer compiles against the installed tt-metal throws on EVERY
// call, and a failed JIT build is not cheap -- so latch an entry point off after its
// first failure rather than paying for it once per node. (As of tt-metal v0.78-dev the
// realize and eltwise kernels build; the norm kernels do not.)
std::atomic<bool> g_off_realize{false}, g_off_bin{false}, g_off_norm{false};

// JIT failures arrive as a multi-page compiler log; keep the histogram readable.
std::string brief(const std::string& msg) {
    const size_t nl = msg.find('\n');
    std::string first = nl == std::string::npos ? msg : msg.substr(0, nl);
    if (first.size() > 160) first.resize(160);
    return first;
}

// Reject reasons are unbounded strings coming out of ttprm's planner; keep the
// histogram readable by tagging them with the call site.
void reject(const char* site, const std::string& why) { g_stats.bump(std::string(site) + ": " + why); }
void accept(const char* site, const std::string& detail) {
    g_stats.bump(std::string(site) + ": *ACCEPT*  " + detail);
}

// ---------------------------------------------------------------------------
// ggml side: fold a view chain into (base tensor, offset, rows, cols, row_stride)
// over the base's FLAT ELEMENT space. That is exactly the map ttprm accepts.
// ---------------------------------------------------------------------------

struct chain_t {
    const ggml_tensor* base = nullptr;
    const ttnn::Tensor* tt = nullptr;
    int64_t offset = 0;      // elements into the base
    int64_t rows = 0;        // presented rows (ne1*ne2*ne3, collapsed)
    int64_t cols = 0;        // presented lanes (ne0)
    int64_t row_stride = 0;  // base elements per presented row; 0 when rows == 1
    int64_t base_rows = 0;   // the flat grid the View anchors on
    int64_t base_cols = 0;
    bool padded_anchor = false;  // base has an interior-padded axis
    // Non-zero grp_P marks a 2-level row map, k(r) = grp_A*(r/grp_P) + grp_B*(r%grp_P),
    // in units of grp_S_row elements. row_stride is then unused.
    int64_t grp_P = 0, grp_A = 0, grp_B = 0, grp_S_row = 0;
};

// Compact shape signature for the histogram, so the accepted views can be told
// apart at a glance.
std::string describe(const chain_t& c) {
    return "[" + std::to_string(c.rows) + "x" + std::to_string(c.cols) + "]" +
           " off=" + std::to_string(c.offset) +
           (c.grp_P ? " grp(P=" + std::to_string(c.grp_P) + ",A=" + std::to_string(c.grp_A) +
                      ",B=" + std::to_string(c.grp_B) + ",S=" + std::to_string(c.grp_S_row) + ")"
                    : " srow=" + std::to_string(c.row_stride)) +
           " src=[" + std::to_string(c.base_rows) + "x" + std::to_string(c.base_cols) + "]" +
           (c.padded_anchor ? " padded-anchor" : " dense-anchor");
}

// Rejects that happen once the chain is known carry its shape, so the histogram says
// WHICH tensors a reason is blocking rather than just how many.
void reject(const char* site, const std::string& why, const chain_t& c) {
    g_stats.bump(std::string(site) + ": " + why + "  " + describe(c));
}

// A node's raw ggml geometry, for rejects that fire before any grid is known. Strides are
// printed in ELEMENTS so they can be compared against ne directly: a chain collapses to one
// row stride exactly when nb[a]/ts == ne[1..a-1] product * nb[1]/ts for every live axis.
std::string describe_node(const ggml_tensor* t) {
    const int64_t ts = (int64_t)ggml_type_size(t->type);
    std::string s = "ne=[";
    for (int i = 0; i < GGML_MAX_DIMS; i++)
        s += std::to_string(t->ne[i]) + (i + 1 < GGML_MAX_DIMS ? "," : "]");
    s += " nb/ts=[";
    for (int i = 0; i < GGML_MAX_DIMS; i++)
        s += std::to_string(ts > 0 ? (int64_t)t->nb[i] / ts : -1) + (i + 1 < GGML_MAX_DIMS ? "," : "]");
    return s;
}

// Collapse (ne, nb) into [rows, cols] + a single row stride.
// The collapsed row index is r = i3*ne2*ne1 + i2*ne1 + i1, so the map is affine
// only when each axis' stride equals (rows collapsed so far) * row_stride.
bool fold_affine(const ggml_tensor* t, chain_t& c) {
    const int64_t ts = (int64_t)ggml_type_size(t->type);
    if (ts <= 0 || (int64_t)t->nb[0] != ts) return false;  // lane stride must be 1

    c.cols = t->ne[0];
    c.rows = 1;
    c.row_stride = 0;
    c.grp_P = c.grp_A = c.grp_B = c.grp_S_row = 0;

    // Live outer axes, innermost first. The collapsed row index is
    // r = ... + i2*ne1 + i1, so axis `a`'s extent multiplies everything inside it.
    struct axis { int64_t n, s; };
    std::vector<axis> live;
    for (int a = 1; a < GGML_MAX_DIMS; a++) {
        if (t->ne[a] == 1) continue;                       // contributes nothing to r
        if ((int64_t)t->nb[a] % ts != 0) return false;
        live.push_back({t->ne[a], (int64_t)t->nb[a] / ts});
        c.rows *= t->ne[a];
    }
    if (live.empty()) return true;                          // single row

    bool nested = true;
    int64_t acc = live[0].n;
    for (size_t i = 1; i < live.size(); i++) {
        if (live[i].s != acc * live[0].s) { nested = false; break; }
        acc *= live[i].n;
    }
    if (nested) { c.row_stride = live[0].s; return true; }  // one uniform row stride

    // Not nested. Exactly two live axes is the 2-level map the device can still walk:
    //     k(r) = live[1].s * (r / live[0].n) + live[0].s * (r % live[0].n)
    // Three independent strides would need a third level, which GROUPED does not have.
    if (live.size() != 2) return false;
    const int64_t B = live[0].s, A = live[1].s;
    if (A <= 0 || B <= 0) return false;
    // Both effective strides must be face-aligned, else an output face-row straddles a
    // source face; their gcd is then face-aligned too and is the finest usable row unit.
    if (A % ttprm::FACE != 0 || B % ttprm::FACE != 0) return false;
    const int64_t S = std::gcd(A, B);
    c.grp_S_row = S; c.grp_A = A / S; c.grp_B = B / S; c.grp_P = live[0].n;
    return true;
}

// Walk down src[0] through VIEW/RESHAPE to the first materialized tensor,
// accumulating view_offs. This deliberately mirrors realize_ggml_view_impl():
// the parent comes from src[0], NOT view_src, because ggml collapses view_src
// past in-place ops while this backend writes in-place results into a fresh
// tensor that IS src[0].
bool resolve_base(const ggml_tensor* t, chain_t& c) {
    int64_t off_bytes = 0;
    const ggml_tensor* cur = t;
    for (int guard = 0; guard < 64; guard++) {
        if (cur->op != GGML_OP_VIEW && cur->op != GGML_OP_RESHAPE) {
            c.base = cur;
            c.tt = ggml_metalium_materialized_tensor(cur);
            if (c.tt == nullptr) return false;
            if ((int64_t)ggml_type_size(t->type) <= 0) return false;
            if (off_bytes % (int64_t)ggml_type_size(t->type) != 0) return false;
            c.offset = off_bytes / (int64_t)ggml_type_size(t->type);
            return true;
        }
        if (cur->op == GGML_OP_VIEW) off_bytes += (int64_t)cur->view_offs;
        if (cur->src[0] == nullptr) return false;
        cur = cur->src[0];
        // Element sizes must agree for the byte offset to fold into one element count.
        if (cur->type != t->type) return false;
    }
    return false;  // pathological chain depth
}

// ---------------------------------------------------------------------------
// ttprm side: anchor a View on the base tensor and cut the chain's rectangle out.
//
// TTNN pads the last two dims of the rank-4 shape, so what ttprm anchors on is a PHYSICAL
// grid: each (ne3,ne2) batch block occupies pad(ne1) rows of pad(ne0) lanes, of which only
// ne1 x ne0 are live. Both anchors View::View can pick index that grid by physical row with
// a pad(ne0) row stride -- so all the arithmetic below is in PHYSICAL rows, which is what
// lets a padded base be used at all rather than rejected.
// ---------------------------------------------------------------------------

struct base_grid {
    int64_t width = 0;       // physical lane width          == pad(ne0)
    int64_t cols = 0;        // live lanes                   == ne0
    int64_t blk_rows = 0;    // physical rows per batch block == pad(ne1)
    int64_t blk_live = 0;    // live rows per batch block     == ne1
    int64_t blocks = 0;      // ne2*ne3
    int64_t anchor_rows = 0; // extent of the anchor's row axis
    int64_t anchor_cols = 0; // extent of the anchor's lane axis
    bool dense_packed = false;
};

std::optional<base_grid> grid_of(const ttnn::Tensor& t) {
    const auto& ls = t.logical_shape();
    const auto& ps = t.padded_shape();
    const size_t rank = ls.rank();
    if (rank < 2 || rank != ps.rank()) return std::nullopt;

    base_grid g;
    g.width    = (int64_t)ps[rank - 1];
    g.cols     = (int64_t)ls[rank - 1];
    g.blk_rows = (int64_t)ps[rank - 2];
    g.blk_live = (int64_t)ls[rank - 2];
    g.blocks   = 1;
    for (size_t i = 0; i + 2 < rank; i++) {
        if ((int64_t)ls[i] != (int64_t)ps[i]) return std::nullopt;  // batch dims are never padded
        g.blocks *= (int64_t)ls[i];
    }
    if (g.cols <= 0 || g.blk_live <= 0 || g.blocks <= 0) return std::nullopt;

    // Mirror View::View's anchor choice, because it decides the row axis' extent.
    // Either way the row INDEX is the physical row: a dense-packed anchor only happens
    // when there is no row padding (or a single block), where the two coincide.
    g.dense_packed = true;
    for (size_t a = 1; a + 1 < rank; a++)
        if ((int64_t)ls[a] != (int64_t)ps[a]) g.dense_packed = false;
    g.anchor_rows = g.dense_packed ? g.blocks * g.blk_live : g.blocks * g.blk_rows;
    g.anchor_cols = g.dense_packed ? g.cols : g.width;
    return g;
}

// Physical row holding ggml row r.
int64_t phys_row(const base_grid& g, int64_t r) {
    return (r / g.blk_live) * g.blk_rows + (r % g.blk_live);
}

// Path 1 -- the view takes a lane range out of whole base rows. This is the only path that
// survives a padded base, because it never assumes consecutive ggml rows are adjacent in
// the anchor's flat space; it walks physical rows with a stride instead.
std::optional<std::vector<ttprm::View::Range>> row_aligned_slice(const base_grid& g, const chain_t& c) {
    const int64_t n0 = g.cols;
    const int64_t c0 = c.offset % n0;
    if (c0 + c.cols > n0) return std::nullopt;      // a presented row must stay inside one base row
    const int64_t r0 = c.offset / n0;
    if (r0 < 0) return std::nullopt;

    int64_t gstep = 1;
    if (c.rows > 1) {
        if (c.row_stride % n0 != 0) return std::nullopt;   // rows do not start on base-row boundaries
        gstep = c.row_stride / n0;
        if (gstep <= 0) return std::nullopt;
    }
    if (r0 + (c.rows - 1) * gstep >= g.blocks * g.blk_live) return std::nullopt;

    // ggml row -> physical row is affine only when row padding cannot bite. Three cases
    // where it provably cannot; anything else needs a 2-level (GROUPED) row map.
    const int64_t p0 = phys_row(g, r0);
    int64_t pstep = 1;
    if (c.rows > 1) {
        if (g.blk_live == g.blk_rows)                          pstep = gstep;
        else if (gstep % g.blk_live == 0)                      pstep = (gstep / g.blk_live) * g.blk_rows;
        else if (r0 % g.blk_live + (c.rows - 1) * gstep < g.blk_live) pstep = gstep;
        else return std::nullopt;
        // Cheap insurance against the case analysis above being wrong.
        if (phys_row(g, r0 + (c.rows - 1) * gstep) != p0 + (c.rows - 1) * pstep) return std::nullopt;
    }
    if (p0 + (c.rows - 1) * pstep >= g.anchor_rows) return std::nullopt;

    return std::vector<ttprm::View::Range>{
        {p0, p0 + (c.rows - 1) * pstep + 1, pstep},
        {c0, c0 + c.cols, 1}};
}

// Cut the chain's [rows, cols] rectangle out of the base's anchor grid.
std::optional<ttprm::View> build_view(chain_t& c, const char* site) {
    if (c.tt->dtype() != tt::tt_metal::DataType::BFLOAT16) { reject(site, "base not bf16"); return std::nullopt; }
    if (c.tt->layout() != tt::tt_metal::Layout::TILE)      { reject(site, "base not TILE"); return std::nullopt; }
    if (c.tt->storage_type() != tt::tt_metal::StorageType::DEVICE) { reject(site, "base not on device"); return std::nullopt; }

    const auto g = grid_of(*c.tt);
    if (!g) { reject(site, "base tensor grid is not describable"); return std::nullopt; }
    c.base_rows = g->anchor_rows;
    c.base_cols = g->anchor_cols;
    c.padded_anchor = (g->cols != g->width) || (g->blk_live != g->blk_rows);
    if (c.rows <= 0 || c.cols <= 0) { reject(site, "empty view", c); return std::nullopt; }

    // A 2-level row map is handed to ttprm directly: LayoutView is affine by construction,
    // so the builder verbs cannot express it. It reads the base's flat element space, which
    // only means anything on a fully dense base.
    if (c.grp_P != 0) {
        if (g->width != g->cols || (g->blk_live != g->blk_rows && g->blocks != 1)) {
            reject(site, "2-level row map over a padded base (would need a third level)", c);
            return std::nullopt;
        }
        if (c.offset % ttprm::FACE != 0 || c.cols % ttprm::FACE != 0) {
            reject(site, "2-level row map is not face-aligned", c);
            return std::nullopt;
        }
        return ttprm::View::grouped(*c.tt, c.rows, c.cols, c.offset,
                                    c.grp_S_row, c.grp_P, c.grp_A, c.grp_B);
    }

    ttprm::View v = ttprm::view_of(*c.tt);
    if (auto ranges = row_aligned_slice(*g, c)) return v.slice(*ranges);

    // Path 2 -- the view re-cuts the flat element stream (a reshape / unflatten, where a
    // presented row spans several base rows or part of one). That only means anything when
    // nothing separates consecutive ggml elements in the anchor: no lane padding, and no
    // row padding between blocks.
    if (g->width != g->cols || (g->blk_live != g->blk_rows && g->blocks != 1)) {
        reject(site, "padded base and the view is not row-aligned (needs a 2-level row map)", c);
        return std::nullopt;
    }
    const int64_t total = g->anchor_rows * g->anchor_cols;
    std::vector<int64_t> widths;
    if (c.rows > 1) widths.push_back(c.row_stride);
    else { widths.push_back(g->anchor_cols); widths.push_back(c.cols); }

    for (const int64_t W : widths) {
        if (W < c.cols || W <= 0 || total % W != 0) continue;
        const int64_t r0 = c.offset / W, c0 = c.offset % W;
        if (c0 + c.cols > W) continue;
        if (r0 < 0 || r0 + c.rows > total / W) continue;
        return v.reshape({total / W, W}).slice({{r0, r0 + c.rows, 1}, {c0, c0 + c.cols, 1}});
    }
    reject(site, "view rectangle does not tile the base grid (row stride / offset)", c);
    return std::nullopt;
}

// Full translation for one ggml tensor.
std::optional<ttprm::View> view_for(const ggml_tensor* t, chain_t& c, const char* site) {
    if (!resolve_base(t, c)) {
        reject(site, "no materialized base in the src[0] chain  " + describe_node(t));
        return std::nullopt;
    }
    if (!ggml_is_contiguous(c.base)) {
        reject(site, "base is not contiguous  " + describe_node(c.base));
        return std::nullopt;
    }
    if (!fold_affine(t, c)) {
        // The interesting case: a second, independent outer stride is the 2-level
        // (GROUPED) row map k(r) = A*(r/P) + B*(r%P). The shapes say whether that is what
        // these are, and whether A/B are whole multiples of the source row width.
        reject(site, "ggml strides do not collapse to one row stride  " + describe_node(t) +
                     " base " + describe_node(c.base));
        return std::nullopt;
    }
    return build_view(c, site);
}

// ---------------------------------------------------------------------------
// Result plumbing.
// ---------------------------------------------------------------------------

// ttprm sizes the tensor it allocates itself as rank-2 [rows, cols]. When the ggml
// node has trivial batch dims that rank-2 shape already IS the ggml shape (leading
// ones), so nothing needs relabelling. Otherwise we must NOT hand back the rank-2
// tensor and ttnn::reshape it to rank 4: that is not the metadata-only relabel the
// page ordering suggests, and it silently yields wrong data. Pre-allocate the real
// rank-4 tensor instead and BIND it as ttprm's output view, so the writer scatters
// straight into it -- no relabel, and one fewer allocation than the rank-2 route.
// Can the node's rank-4 shape be handed to ttprm as a bound output view?
//
// view_of(dst) collapses the tensor to [ne3*ne2*ne1, ne0] and must present the same grid
// the input view does. Only a padded ROW count breaks that: View::View inspects the rank-4
// shape's middle axes, so a padded ne1 drops it to the padded anchor and prelower then
// rejects on a row-count mismatch. A padded ne0 is fine -- it is absorbed into the layout
// stride as pcols, and both endpoints end up with the same Co/Co_pad.
// ttprm scatters a bound destination through a 2-level row map when ttnn pads its row
// dim, so any rank-4 tile tensor can now be bound. The rank-2 + rank-extension route is
// kept for the shapes where it is simply cheaper (no grouped addressing needed).
bool prefer_rank_extension(const ggml_tensor* node) {
    return node->ne[2] * node->ne[3] == 1 && node->ne[1] % ttprm::TILE != 0;
}
bool output_shape_ok(const ggml_tensor*) { return true; }

struct out_plan {
    bool ok = false;
    std::optional<ttnn::Tensor> bound;  // empty -> let ttprm allocate the rank-2 result
};

out_plan plan_output(const ggml_tensor* node, const ttnn::Tensor& like, const char* site) {
    out_plan p;
    if (prefer_rank_extension(node)) {
        // ttprm's rank-2 [rows, cols] result already IS this shape modulo leading ones,
        // and a pure rank extension leaves the tiling untouched.
        p.ok = true;
        return p;
    }

    std::array<uint32_t, GGML_MAX_DIMS> dims;
    for (int i = 0; i < GGML_MAX_DIMS; i++) dims[i] = (uint32_t)node->ne[GGML_MAX_DIMS - i - 1];
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape(dims),
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16,
                                   tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                                   tt::tt_metal::MemoryConfig{}));
    p.bound = ttnn::create_device_tensor(spec, like.device());
    p.ok = true;
    return p;
}

// Give the result the shape the backend expects. A bound output already has it. An
// unbound (rank-2) result only needs the leading ones put back, which leaves the last
// two dims -- and therefore the physical tiling -- untouched, so ttnn::reshape is a
// pure relabel here. A reshape that CHANGES the last two dims is not, so it is refused
// in plan_output() rather than attempted.
bool carries_ggml_shape(const ttnn::Shape& shape, const ggml_tensor* node) {
    if (shape.rank() > GGML_MAX_DIMS) return false;
    for (size_t i = 0; i < shape.rank(); i++)
        if (node->ne[GGML_MAX_DIMS - i - 1] != (int64_t)shape[i]) return false;
    for (size_t i = shape.rank(); i < GGML_MAX_DIMS; i++)
        if (node->ne[GGML_MAX_DIMS - i - 1] != 1) return false;
    return true;
}

std::shared_ptr<ttnn::Tensor> label_as(ttnn::Tensor res, const ggml_tensor* node, const char* site) {
    const ttnn::Shape& shape = res.logical_shape();
    if (carries_ggml_shape(shape, node)) return std::make_shared<ttnn::Tensor>(std::move(res));

    if (shape.rank() == 2 && node->ne[2] * node->ne[3] == 1 &&
        (int64_t)shape[0] == node->ne[1] && (int64_t)shape[1] == node->ne[0]) {
        const std::array<uint32_t, GGML_MAX_DIMS> dims{
            1u, 1u, (uint32_t)node->ne[1], (uint32_t)node->ne[0]};
        return std::make_shared<ttnn::Tensor>(ttnn::reshape(res, ttnn::Shape(dims)));
    }
    reject(site, "result shape != node shape");
    return nullptr;
}

// ggml broadcasts src1 into src0 by repeating whole trailing axes. ttprm's
// classify_operand() re-feeds a smaller operand as a resident group, which is the
// SAME thing only when src1's shape is a prefix of src0's followed by ones --
// e.g. [C,N,1] into [C,N,B]. For [C,1,B] into [C,N,B] the tile counts still
// divide, so ttprm would accept and silently repeat the wrong rows. Gate here.
bool broadcast_is_group_shaped(const ggml_tensor* a, const ggml_tensor* b) {
    if (a->ne[0] != b->ne[0]) return false;   // ttprm requires equal lane widths
    bool trailing_ones = false;
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (trailing_ones) { if (b->ne[i] != 1) return false; continue; }
        if (b->ne[i] == a->ne[i]) continue;
        if (b->ne[i] == 1) { trailing_ones = true; continue; }
        return false;
    }
    return true;
}

// A node this backend does not materialize on its own, so whoever consumes it pays to
// realize it first. Those are the only operands worth handing to ttprm: for one that
// already has a device tensor, TTNN's own op is a single well-tuned dispatch and
// routing it through ttprm buys nothing. Note an in-place op's result carries a
// view_src but DOES have its own tensor, hence the second check.
bool is_viewed(const ggml_tensor* t) {
    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE) return true;
    return t->view_src != nullptr && ggml_metalium_materialized_tensor(t) == nullptr;
}

}  // namespace

bool enabled() {
    static const bool on = parse_env_default("GGML_METALIUM_TTPRM", true);
    return on;
}

std::shared_ptr<ttnn::Tensor> realize_view(const ggml_tensor* node) {
    static constexpr const char* site = "realize";
    if (!enabled()) return nullptr;
    if (g_off_realize.load(std::memory_order_relaxed)) return nullptr;
    if (!output_shape_ok(node)) { reject(site, "result rows are not tile-aligned and it is not a rank extension"); return nullptr; }

    chain_t c;
    auto v = view_for(node, c, site);
    if (!v) return nullptr;

    try {
        out_plan op = plan_output(node, *c.tt, site);
        if (!op.ok) return nullptr;
        std::optional<ttprm::View> out_view;
        if (op.bound) out_view = ttprm::view_of(*op.bound);

        auto res = ttprm::realize(*v, out_view);
        if (!res) { reject(site, res.error(), c); return nullptr; }
        accept(site, describe(c));
        return label_as(res.value(), node, site);
    } catch (const std::exception& e) {
        g_off_realize.store(true, std::memory_order_relaxed);
        reject(site, "disabled after a device-side failure: " + brief(e.what()));
        return nullptr;
    }
}

std::shared_ptr<ttnn::Tensor> bin_op(const ggml_tensor* dst, ggml_op op) {
    static constexpr const char* site = "bin_op";
    if (!enabled()) return nullptr;
    if (g_off_bin.load(std::memory_order_relaxed)) return nullptr;
    if (op != GGML_OP_ADD && op != GGML_OP_SUB && op != GGML_OP_MUL) return nullptr;  // no ttprm divide

    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    if (!is_viewed(src0) && !is_viewed(src1)) return nullptr;

    if (!ggml_are_same_shape(dst, src0))            { reject(site, "dst shape != src0 shape"); return nullptr; }
    if (!broadcast_is_group_shaped(src0, src1)) {
        reject(site, "src1 broadcast is not a trailing-axis repeat  a=" + describe_node(src0) +
                     " b=" + describe_node(src1));
        return nullptr;
    }
    if (!output_shape_ok(dst))                      { reject(site, "result rows are not tile-aligned and it is not a rank extension"); return nullptr; }

    chain_t ca, cb;
    auto va = view_for(src0, ca, site);
    if (!va) return nullptr;
    auto vb = view_for(src1, cb, site);
    if (!vb) return nullptr;

    try {
        out_plan op_out = plan_output(dst, *ca.tt, site);
        if (!op_out.ok) return nullptr;
        std::optional<ttprm::View> out_view;
        if (op_out.bound) out_view = ttprm::view_of(*op_out.bound);

        ttprm::Result<ttnn::Tensor> res = ttprm::Result<ttnn::Tensor>::err("unreachable");
        switch (op) {
            case GGML_OP_ADD: res = ttprm::add(*va, *vb, out_view); break;
            case GGML_OP_SUB: res = ttprm::sub(*va, *vb, out_view); break;
            case GGML_OP_MUL: res = ttprm::mul(*va, *vb, out_view); break;
            default: return nullptr;
        }
        if (!res) { reject(site, res.error() + "  b=" + describe(cb), ca); return nullptr; }
        accept(site, describe(ca) + "  b=" + describe(cb));
        return label_as(res.value(), dst, site);
    } catch (const std::exception& e) {
        g_off_bin.store(true, std::memory_order_relaxed);
        reject(site, "disabled after a device-side failure: " + brief(e.what()));
        return nullptr;
    }
}

std::shared_ptr<ttnn::Tensor> norm(const ggml_tensor* dst, bool rms, float eps) {
    static constexpr const char* site = "norm";
    if (!enabled()) return nullptr;
    if (g_off_norm.load(std::memory_order_relaxed)) return nullptr;

    const ggml_tensor* src0 = dst->src[0];
    if (!is_viewed(src0)) return nullptr;
    if (!ggml_are_same_shape(dst, src0))  { reject(site, "dst shape != src0 shape"); return nullptr; }
    if (!output_shape_ok(dst))            { reject(site, "result rows are not tile-aligned and it is not a rank extension"); return nullptr; }

    chain_t c;
    auto v = view_for(src0, c, site);
    if (!v) return nullptr;

    try {
        out_plan op_out = plan_output(dst, *c.tt, site);
        if (!op_out.ok) return nullptr;
        std::optional<ttprm::View> out_view;
        if (op_out.bound) out_view = ttprm::view_of(*op_out.bound);

        // ggml's RMS_NORM/NORM carry no affine parameters; the weight, when a model has
        // one, is a separate GGML_OP_MUL node.
        auto res = rms ? ttprm::rms_norm(*v, eps, out_view)
                       : ttprm::layer_norm(*v, /*gamma=*/nullptr, /*beta=*/nullptr, eps, out_view);
        if (!res) { reject(site, res.error(), c); return nullptr; }
        accept(site, describe(c));
        return label_as(res.value(), dst, site);
    } catch (const std::exception& e) {
        g_off_norm.store(true, std::memory_order_relaxed);
        reject(site, "disabled after a device-side failure: " + brief(e.what()));
        return nullptr;
    }
}

}  // namespace ggml_ttprm

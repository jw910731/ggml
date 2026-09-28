#include "metalium_layout.hpp"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <map>
#include <tuple>

static constexpr int64_t ML_TILE = 32;

ml_grid ml_grid_of(const int64_t * tt_shape, int rank) {
    ml_grid g;
    g.W = rank >= 1 ? tt_shape[rank - 1] : 1;
    g.R = rank >= 2 ? tt_shape[rank - 2] : 1;
    for (int i = 0; i + 2 < rank; i++) {
        g.blocks *= tt_shape[i];
    }
    return g;
}

std::array<int64_t, 4> ml_coarsen(const int64_t * ne, int g) {
    std::array<int64_t, 4> out = {1, 1, 1, 1};
    for (int d = 0; d < g; d++) {
        out[3] *= ne[d];
    }
    for (int d = g, k = 2; d < GGML_MAX_DIMS; d++, k--) {
        out[k] = ne[d];
    }
    return out;
}

// reshape.cpp: a TILE reshape is a view when the last dim is kept and the second-to-last is either
// kept or tile-aligned on both sides
bool ml_reshape_is_view(const std::array<int64_t, 4> & from, const std::array<int64_t, 4> & to) {
    return from[3] == to[3] && (from[2] == to[2] || (from[2] % ML_TILE == 0 && to[2] % ML_TILE == 0));
}

static bool ml_is_view_op(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

const char * ml_resolve_chain(const ggml_tensor * t, ml_access & acc) {
    if (ggml_blck_size(t->type) != 1) {
        return "block-quantized type";
    }
    const int64_t ts = (int64_t) ggml_type_size(t->type);

    // The parent comes from src[0], not view_src: ggml collapses view_src past in-place ops, whose
    // results the backend keeps in the in-place node itself. view_offs is cumulative down to
    // view_src, so the offset is the sum of each VIEW's own one.
    int64_t             off_bytes = 0;
    const ggml_tensor * cur       = t;
    for (int depth = 0; ml_is_view_op(cur); depth++) {
        if (depth >= 64 || cur->src[0] == nullptr) {
            return "view chain has no base";
        }
        if (cur->op == GGML_OP_VIEW) {
            size_t local = 0;
            memcpy(&local, cur->op_params, sizeof(local));
            off_bytes += (int64_t) local;
        }
        cur = cur->src[0];
        if (cur->type != t->type) {
            return "type changes along the chain";
        }
    }
    if (!ggml_is_contiguous(cur)) {
        return "base is not contiguous";
    }
    if (t->data != nullptr && cur->data != nullptr) {
        GGML_ASSERT((const char *) t->data - (const char *) cur->data == off_bytes);
    }
    if (off_bytes % ts != 0) {
        return "offset is not a whole element";
    }

    acc.base   = cur;
    acc.offset = off_bytes / ts;
    int64_t last = acc.offset;
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        if ((int64_t) t->nb[d] % ts != 0) {
            return "stride is not a whole element";
        }
        acc.ne[d]     = t->ne[d];
        acc.ne_src[d] = t->ne[d];
        acc.st[d]     = (int64_t) t->nb[d] / ts;
        last += (t->ne[d] - 1) * acc.st[d];
    }
    if (acc.offset < 0 || last >= ggml_nelements(cur)) {
        return "reads outside the base";
    }
    return nullptr;
}

const char * ml_resolve_repeat(const ggml_tensor * dst, ml_access & acc) {
    const ggml_tensor * src = dst->src[0];
    if (const char * why = ml_resolve_chain(src, acc)) {
        return why;
    }
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        if (src->ne[d] <= 0 || dst->ne[d] % src->ne[d] != 0) {
            return "not a whole repeat";
        }
        acc.ne[d]     = dst->ne[d];
        acc.ne_src[d] = src->ne[d];
        if (src->ne[d] == 1) {
            acc.st[d] = 0;
        }
    }
    return nullptr;
}

bool ml_plan::identity() const {
    if (planes.size() != 1) {
        return false;
    }
    const ml_plane & p = planes[0];
    return p.op == ml_lane_op::identity && p.blk0 == 0 && p.nblk == in.blocks && p.row0 == 0 && p.nrow == in.R;
}

const char * ml_plan_row_local(const ml_access & acc, int g_out, const ml_grid & in, ml_plan & plan) {
    if (g_out < 1 || g_out > 3) {
        return "lane dim count out of range";
    }
    if (in.W <= 0 || in.R <= 0 || in.blocks <= 0) {
        return "empty base";
    }
    plan           = ml_plan{};
    plan.in        = in;
    plan.out_shape = ml_coarsen(acc.ne, g_out);
    plan.W_out     = plan.out_shape[3];
    if (plan.W_out <= 0 || plan.W_out > (1 << 20)) {
        return "lane count out of range";
    }

    // Output rows beyond the lanes are the dims above g_out, flattened. They must read consecutive
    // base rows in the same order, except that the outermost may instead pick one plane per index:
    // a lane offset, a block or a row range of its own.
    int     dims[GGML_MAX_DIMS];
    int     n_dims = 0;
    for (int d = g_out; d < GGML_MAX_DIMS; d++) {
        if (acc.ne[d] > 1) {
            dims[n_dims++] = d;
        }
    }
    int     plane  = -1;
    int64_t expect = in.W;
    int64_t n_rows = 1;
    for (int k = 0; k < n_dims; k++) {
        const int d = dims[k];
        if (acc.ne_src[d] == acc.ne[d] && acc.st[d] == expect) {
            expect *= acc.ne[d];
            n_rows *= acc.ne[d];
            continue;
        }
        if (k + 1 != n_dims) {
            return "rows are not consecutive base rows";
        }
        plane = d;
    }
    const int64_t n_planes = plane < 0 ? 1 : acc.ne[plane];
    if (n_planes > 64) {
        return "more than 64 planes";
    }

    for (int64_t q = 0; q < n_planes; q++) {
        const int64_t src_q = plane < 0 ? 0 : q % acc.ne_src[plane];
        if (src_q != q) {
            ml_plane dup = plan.planes[src_q];
            dup.dup_of   = (int) src_q;
            dup.lanes.clear();
            plan.planes.push_back(std::move(dup));
            continue;
        }
        const int64_t off = acc.offset + (plane < 0 ? 0 : q * acc.st[plane]);
        const int64_t r0  = off / in.W;
        const int64_t c0  = off % in.W;

        ml_plane p;
        p.lanes.resize(plan.W_out);
        bool identity = plan.W_out == in.W;
        bool window   = true;
        for (int64_t c = 0; c < plan.W_out; c++) {
            int64_t rem = c;
            int64_t pos = c0;
            for (int d = 0; d < g_out; d++) {
                const int64_t i = rem % acc.ne[d];
                rem /= acc.ne[d];
                pos += (i % acc.ne_src[d]) * acc.st[d];
            }
            if (pos >= in.W) {
                return "lanes cross base rows";
            }
            p.lanes[c] = (int32_t) pos;
            identity   = identity && pos == c;
            window     = window && pos == p.lanes[0] + c;
        }
        p.op = identity ? ml_lane_op::identity :
               (window && p.lanes[0] % ML_TILE == 0) ? ml_lane_op::window : ml_lane_op::gather;

        const int64_t R = in.R;
        if (r0 % R == 0 && n_rows % R == 0) {
            p.blk0 = r0 / R;
            p.nblk = n_rows / R;
            p.row0 = 0;
            p.nrow = R;
        } else if (r0 / R == (r0 + n_rows - 1) / R) {
            if (r0 % R % ML_TILE != 0) {
                return "row offset is not tile-aligned";
            }
            p.blk0 = r0 / R;
            p.nblk = 1;
            p.row0 = r0 % R;
            p.nrow = n_rows;
        } else {
            return "rows are neither whole blocks nor inside one block";
        }
        if (p.blk0 + p.nblk > in.blocks) {
            return "rows outside the base";
        }
        if (!plan.planes.empty() && (p.nblk != plan.planes[0].nblk || p.nrow != plan.planes[0].nrow)) {
            return "planes read different row counts";
        }
        plan.planes.push_back(std::move(p));
    }

    const ml_plane &             p0  = plan.planes[0];
    const std::array<int64_t, 4> cat = {1, n_planes * p0.nblk, p0.nrow, plan.W_out};
    GGML_ASSERT(cat[1] * cat[2] * cat[3] ==
                plan.out_shape[0] * plan.out_shape[1] * plan.out_shape[2] * plan.out_shape[3]);
    if (!ml_reshape_is_view(cat, plan.out_shape)) {
        return "result needs a data-moving reshape";
    }
    return nullptr;
}

std::vector<ml_run> ml_classify_runs(const ml_plan & plan) {
    std::vector<ml_run> runs;
    const int64_t       n_tiles = (plan.W_out + ML_TILE - 1) / ML_TILE;
    for (size_t q = 0; q < plan.planes.size(); q++) {
        const ml_plane & p = plan.planes[q];
        if (p.dup_of >= 0) {
            continue;
        }
        for (int64_t t = 0; t < n_tiles; t++) {
            const int64_t c0    = t * ML_TILE;
            const int64_t valid = std::min(ML_TILE, plan.W_out - c0);
            const int64_t s0    = p.lanes[c0];
            // A partial last tile only copies whole when the base tile ends at the same lane
            bool page = s0 >= 0 && s0 % ML_TILE == 0 && (valid == ML_TILE || s0 + valid == plan.in.W);
            for (int64_t c = 0; page && c < valid; c++) {
                page = p.lanes[c0 + c] == s0 + c;
            }
            runs.push_back({(int) q, t, page, page ? s0 / ML_TILE : -1});
        }
    }
    return runs;
}

std::vector<float> ml_apply_plan_host(const ml_plan & plan, const std::vector<float> & base) {
    std::vector<float>  out;
    std::vector<size_t> starts;
    for (const ml_plane & p : plan.planes) {
        const size_t start = out.size();
        starts.push_back(start);
        if (p.dup_of >= 0) {
            const size_t from = starts[p.dup_of];
            const size_t n    = (size_t) (p.nblk * p.nrow * plan.W_out);
            for (size_t i = 0; i < n; i++) {
                out.push_back(out[from + i]);
            }
            continue;
        }
        for (int64_t b = 0; b < p.nblk; b++) {
            for (int64_t r = 0; r < p.nrow; r++) {
                const int64_t row = (p.blk0 + b) * plan.in.R + p.row0 + r;
                for (int64_t c = 0; c < plan.W_out; c++) {
                    const int32_t lane = p.lanes[c];
                    out.push_back(lane < 0 ? 0.0f : base[(size_t) (row * plan.in.W + lane)]);
                }
            }
        }
    }
    return out;
}

// A unit's slots, zero slot included, must stay addressable by 16-bit halfword offsets
static constexpr int64_t ML_REMAP_MAX_SRC   = 32;
static constexpr size_t  ML_REMAP_MAX_WORDS = 16384;

// Halfword offset of element (r, c) in a tile of four 16x16 faces
static int64_t ml_face_offset(int64_t r, int64_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

namespace {
struct ml_record {
    ml_run               run;
    std::vector<int64_t> src;  // base lane tiles read, ascending
};
}  // namespace

const char * ml_build_remap(const ml_plan & plan, int64_t workers, ml_remap & out) {
    out                  = ml_remap{};
    const ml_plane & p0  = plan.planes[0];
    out.nblk             = p0.nblk;
    out.nrow             = p0.nrow;
    const int64_t ct_in  = (plan.in.W + ML_TILE - 1) / ML_TILE;
    const int64_t rt_out = (p0.nrow + ML_TILE - 1) / ML_TILE;

    std::vector<std::vector<uint32_t>> dsts(plan.planes.size());
    for (size_t q = 0; q < plan.planes.size(); q++) {
        GGML_ASSERT(plan.planes[q].row0 % ML_TILE == 0);
        dsts[plan.planes[q].dup_of < 0 ? q : (size_t) plan.planes[q].dup_of].push_back((uint32_t) q);
    }

    std::vector<ml_record> recs;
    for (const ml_run & run : ml_classify_runs(plan)) {
        ml_record rec{run, {}};
        if (run.page) {
            rec.src.push_back(run.src_tile);
        } else {
            const ml_plane & p     = plan.planes[run.plane];
            const int64_t    c0    = run.out_tile * ML_TILE;
            const int64_t    valid = std::min(ML_TILE, plan.W_out - c0);
            for (int64_t c = 0; c < valid; c++) {
                if (p.lanes[c0 + c] >= 0) {
                    rec.src.push_back(p.lanes[c0 + c] / ML_TILE);
                }
            }
            std::sort(rec.src.begin(), rec.src.end());
            rec.src.erase(std::unique(rec.src.begin(), rec.src.end()), rec.src.end());
        }
        recs.push_back(std::move(rec));
    }
    if (recs.size() * 18 + plan.planes.size() > ML_REMAP_MAX_WORDS) {
        return "lane table too large";
    }

    // Records of planes that read the same rows share a unit while they share base tiles, so each
    // tile is read once; with too few units for the cores, every record gets a unit of its own.
    struct chunk {
        std::vector<size_t>  recs;
        std::vector<int64_t> src;
    };

    auto make_chunks = [&](bool join) {
        std::vector<chunk>                                      chunks;
        std::map<std::tuple<int64_t, int64_t, int64_t>, size_t> owner;  // (blk0, row0, base tile) -> chunk
        for (size_t i = 0; i < recs.size(); i++) {
            const ml_plane &     p = plan.planes[recs[i].run.plane];
            std::vector<int64_t> merged;
            size_t               at = chunks.size();
            for (size_t k = 0; join && at == chunks.size() && k < recs[i].src.size(); k++) {
                auto it = owner.find({p.blk0, p.row0, recs[i].src[k]});
                if (it == owner.end()) {
                    continue;
                }
                const chunk & ch = chunks[it->second];
                merged.clear();
                std::set_union(ch.src.begin(), ch.src.end(), recs[i].src.begin(), recs[i].src.end(),
                               std::back_inserter(merged));
                if ((int64_t) merged.size() <= ML_REMAP_MAX_SRC) {
                    at = it->second;
                }
            }
            if (at == chunks.size()) {
                chunks.push_back({{i}, recs[i].src});
            } else {
                chunks[at].recs.push_back(i);
                chunks[at].src = std::move(merged);
            }
            for (int64_t tile : recs[i].src) {
                owner.try_emplace({p.blk0, p.row0, tile}, at);
            }
        }
        return chunks;
    };
    std::vector<chunk> chunks = make_chunks(true);
    if ((int64_t) chunks.size() * p0.nblk * rt_out < workers) {
        chunks = make_chunks(false);
    }

    int64_t max_src = 0;
    for (const chunk & ch : chunks) {
        if ((int64_t) ch.src.size() > ML_REMAP_MAX_SRC) {
            return "an output tile reads too many base tiles";
        }
        max_src = std::max(max_src, (int64_t) ch.src.size());
    }
    out.n_chunks        = (int64_t) chunks.size();
    out.slots           = max_src + 1;
    const uint32_t zero = (uint32_t) (max_src * 1024);

    std::vector<uint32_t> & w = out.table;
    w.assign(1 + chunks.size(), 0);
    w[0] = (uint32_t) chunks.size();
    for (size_t c = 0; c < chunks.size(); c++) {
        const chunk &    ch = chunks[c];
        const ml_plane & p  = plan.planes[recs[ch.recs[0]].run.plane];

        auto slot_of = [&](int64_t tile) {
            return (uint32_t) (std::lower_bound(ch.src.begin(), ch.src.end(), tile) - ch.src.begin());
        };
        uint32_t flags = 0;
        for (size_t i : ch.recs) {
            if (recs[i].run.page && plan.in.W % ML_TILE != 0 && recs[i].run.src_tile == ct_in - 1) {
                flags |= ML_REMAP_ZERO_PAD_LANES;
            }
        }
        w[1 + c] = (uint32_t) w.size();
        w.insert(w.end(), {(uint32_t) p.blk0, (uint32_t) (p.row0 / ML_TILE), (uint32_t) ch.src.size(), flags,
                           (uint32_t) ch.recs.size()});
        for (int64_t tile : ch.src) {
            w.push_back((uint32_t) tile);
        }
        for (size_t i : ch.recs) {
            const ml_run &                run = recs[i].run;
            const std::vector<uint32_t> & d   = dsts[run.plane];
            w.push_back((run.page ? 0u : 1u << 31) | (uint32_t) d.size() << 16 | (uint32_t) run.out_tile);
            w.insert(w.end(), d.begin(), d.end());
            if (run.page) {
                w.push_back(slot_of(run.src_tile));
                continue;
            }
            const std::vector<int32_t> & lanes = plan.planes[run.plane].lanes;

            auto source = [&](int64_t c) {
                if (c >= plan.W_out || lanes[c] < 0) {
                    return zero;
                }
                return slot_of(lanes[c] / ML_TILE) * 1024 + (uint32_t) ml_face_offset(0, lanes[c] % ML_TILE);
            };
            for (int64_t k = 0; k < ML_TILE / 2; k++) {
                const int64_t c = run.out_tile * ML_TILE + 2 * k;
                w.push_back(source(c) | source(c + 1) << 16);
            }
        }
    }
    w.resize((w.size() + ML_REMAP_PAGE_WORDS - 1) / ML_REMAP_PAGE_WORDS * ML_REMAP_PAGE_WORDS, 0);
    if (w.size() > ML_REMAP_MAX_WORDS) {
        return "lane table too large";
    }
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t v : w) {
        h = (h ^ v) * 0x100000001b3ull;
    }
    out.hash = h;
    return nullptr;
}

const char * ml_bin_broadcast(const int64_t * dst_ne, const int64_t * src1_ne, int g) {
    bool lanes_equal = true;
    bool lanes_one   = true;
    for (int d = 0; d < g; d++) {
        lanes_equal = lanes_equal && src1_ne[d] == dst_ne[d];
        lanes_one   = lanes_one && src1_ne[d] == 1;
    }
    if (!lanes_equal && !lanes_one) {
        return "src1 broadcasts part of a lane group";
    }
    for (int d = g; d < GGML_MAX_DIMS; d++) {
        if (src1_ne[d] != dst_ne[d] && src1_ne[d] != 1) {
            return "src1 does not broadcast into dst";
        }
    }
    return nullptr;
}

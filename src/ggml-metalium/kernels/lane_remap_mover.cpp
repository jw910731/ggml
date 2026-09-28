#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"

// One of the two data-movement RISCs of a lane_remap core (lane_remap.hpp). The run table layout
// is documented at ml_remap (metalium_layout.hpp). All tiles are bf16 in four 16x16 faces: the
// halfword of (r, c) is at ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15).

namespace {

constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t TABLE_PAGE_BYTES = 1024;
constexpr uint32_t GATHER = 1u << 31;
constexpr uint32_t ZERO_PAD_LANES = 1;

void zero_rows(uint32_t tile, uint32_t from) {
    for (uint32_t r = from; r < 32; r++) {
        tt_l1_ptr uint32_t* row = reinterpret_cast<tt_l1_ptr uint32_t*>(tile) + (r >> 4) * 256 + (r & 15) * 8;
        for (uint32_t i = 0; i < 8; i++) {
            row[i] = 0;
            row[128 + i] = 0;
        }
    }
}

void zero_lanes(uint32_t tile, uint32_t from) {
    tt_l1_ptr uint16_t* t = reinterpret_cast<tt_l1_ptr uint16_t*>(tile);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = from; c < 32; c++) {
            t[((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)] = 0;
        }
    }
}

// Output words 4g..4g + 3 of all 32 rows, word k holding the halfwords at slot offsets lo[k] and
// hi[k] of row 0. Rows step by 16 halfwords within a face and by 512 into the lower faces. The four
// words of a row fill one aligned 16-byte block, so the store queue coalesces them. Loads cost more
// than anything else here, so runs of adjacent sources are loaded a word at a time.
void gather_group(tt_l1_ptr uint32_t* out, const tt_l1_ptr uint16_t* slots, const tt_l1_ptr uint32_t* pairs) {
    uint32_t lo[4], hi[4];
    bool words = (pairs[0] & 1) == 0, dups = true, dup_run = (pairs[0] & 1) == 0;
    for (uint32_t k = 0; k < 4; k++) {
        lo[k] = pairs[k] & 0xffff;
        hi[k] = pairs[k] >> 16;
        words = words && hi[k] == lo[k] + 1 && lo[k] == lo[0] + 2 * k;
        dups = dups && hi[k] == lo[k];
        dup_run = dup_run && hi[k] == lo[k] && lo[k] == lo[0] + k;
    }
    if (words) {
        const tt_l1_ptr uint32_t* a = reinterpret_cast<const tt_l1_ptr uint32_t*>(slots + lo[0]);
        for (uint32_t half = 0; half < 2; half++, a += 256, out += 256) {
#pragma GCC unroll 16
            for (uint32_t r = 0; r < 16; r++) {
                const uint32_t w0 = a[r * 8], w1 = a[r * 8 + 1], w2 = a[r * 8 + 2], w3 = a[r * 8 + 3];
                out[r * 8] = w0;
                out[r * 8 + 1] = w1;
                out[r * 8 + 2] = w2;
                out[r * 8 + 3] = w3;
            }
        }
    } else if (dup_run) {
        const tt_l1_ptr uint32_t* a = reinterpret_cast<const tt_l1_ptr uint32_t*>(slots + lo[0]);
        for (uint32_t half = 0; half < 2; half++, a += 256, out += 256) {
#pragma GCC unroll 16
            for (uint32_t r = 0; r < 16; r++) {
                const uint32_t w0 = a[r * 8], w1 = a[r * 8 + 1];
                out[r * 8] = w0 << 16 | (w0 & 0xffff);
                out[r * 8 + 1] = w0 >> 16 | (w0 & 0xffff0000);
                out[r * 8 + 2] = w1 << 16 | (w1 & 0xffff);
                out[r * 8 + 3] = w1 >> 16 | (w1 & 0xffff0000);
            }
        }
    } else if (dups) {
        const tt_l1_ptr uint16_t *a0 = slots + lo[0], *a1 = slots + lo[1], *a2 = slots + lo[2], *a3 = slots + lo[3];
        for (uint32_t half = 0; half < 2; half++, a0 += 512, a1 += 512, a2 += 512, a3 += 512, out += 256) {
#pragma GCC unroll 16
            for (uint32_t r = 0; r < 16; r++) {
                const uint32_t v0 = a0[r * 16], v1 = a1[r * 16], v2 = a2[r * 16], v3 = a3[r * 16];
                out[r * 8] = v0 | v0 << 16;
                out[r * 8 + 1] = v1 | v1 << 16;
                out[r * 8 + 2] = v2 | v2 << 16;
                out[r * 8 + 3] = v3 | v3 << 16;
            }
        }
    } else {
        const tt_l1_ptr uint16_t *a0 = slots + lo[0], *a1 = slots + lo[1], *a2 = slots + lo[2], *a3 = slots + lo[3];
        const tt_l1_ptr uint16_t *b0 = slots + hi[0], *b1 = slots + hi[1], *b2 = slots + hi[2], *b3 = slots + hi[3];
        for (uint32_t half = 0; half < 2; half++, out += 256) {
#pragma GCC unroll 16
            for (uint32_t r = 0; r < 16; r++) {
                const uint32_t x0 = a0[r * 16], x1 = a1[r * 16], x2 = a2[r * 16], x3 = a3[r * 16];
                const uint32_t y0 = b0[r * 16], y1 = b1[r * 16], y2 = b2[r * 16], y3 = b3[r * 16];
                out[r * 8] = x0 | y0 << 16;
                out[r * 8 + 1] = x1 | y1 << 16;
                out[r * 8 + 2] = x2 | y2 << 16;
                out[r * 8 + 3] = x3 | y3 << 16;
            }
            a0 += 512, a1 += 512, a2 += 512, a3 += 512, b0 += 512, b1 += 512, b2 += 512, b3 += 512;
        }
    }
}

// Two records whose sources differ by one halfword everywhere, e.g. the even and odd lanes of a
// stride-2 split: where a group reads 16 adjacent halfwords, each word loaded feeds both outputs.
void gather_group_pair(
    tt_l1_ptr uint32_t* even, tt_l1_ptr uint32_t* odd, const tt_l1_ptr uint16_t* slots, const tt_l1_ptr uint32_t* pairs) {
    const uint32_t lo = pairs[0] & 0xffff;
    bool split = (lo & 1) == 0;
    for (uint32_t k = 0; k < 4; k++) {
        split = split && pairs[k] == ((lo + 4 * k) | (lo + 4 * k + 2) << 16);
    }
    if (!split) {
        const uint32_t odd_pairs[4] = {pairs[0] + 0x10001, pairs[1] + 0x10001, pairs[2] + 0x10001, pairs[3] + 0x10001};
        gather_group(even, slots, pairs);
        gather_group(odd, slots, odd_pairs);
        return;
    }
    const tt_l1_ptr uint32_t* a = reinterpret_cast<const tt_l1_ptr uint32_t*>(slots + lo);
    for (uint32_t half = 0; half < 2; half++, a += 256, even += 256, odd += 256) {
#pragma GCC unroll 4
        for (uint32_t r = 0; r < 16; r++) {
            const uint32_t w0 = a[r * 8], w1 = a[r * 8 + 1], w2 = a[r * 8 + 2], w3 = a[r * 8 + 3];
            const uint32_t w4 = a[r * 8 + 4], w5 = a[r * 8 + 5], w6 = a[r * 8 + 6], w7 = a[r * 8 + 7];
            even[r * 8] = (w0 & 0xffff) | w1 << 16;
            even[r * 8 + 1] = (w2 & 0xffff) | w3 << 16;
            even[r * 8 + 2] = (w4 & 0xffff) | w5 << 16;
            even[r * 8 + 3] = (w6 & 0xffff) | w7 << 16;
            odd[r * 8] = w0 >> 16 | (w1 & 0xffff0000);
            odd[r * 8 + 1] = w2 >> 16 | (w3 & 0xffff0000);
            odd[r * 8 + 2] = w4 >> 16 | (w5 & 0xffff0000);
            odd[r * 8 + 3] = w6 >> 16 | (w7 & 0xffff0000);
        }
    }
}

// Stores to L1 must have landed before the NoC reads the tile out. The fence sends every queued
// store to L1 whatever order the compiler emitted them in, and L1 serves one RISC's requests in
// order, so a blocking load of any L1 word returns only after all of them have landed.
void drain(uint32_t l1_word) {
    asm volatile("fence" ::: "memory");
    (void)ckernel::load_blocking(reinterpret_cast<volatile uint32_t*>(l1_word));
}

} // namespace

void kernel_main() {
    const uint32_t base_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t table_bytes = get_arg_val<uint32_t>(3);
    const uint32_t unit_begin = get_arg_val<uint32_t>(4);
    const uint32_t unit_end = get_arg_val<uint32_t>(5);
    const uint32_t nblk = get_arg_val<uint32_t>(6);
    const uint32_t rt_out = get_arg_val<uint32_t>(7);
    const uint32_t ct_out = get_arg_val<uint32_t>(8);
    const uint32_t rt_in = get_arg_val<uint32_t>(9);
    const uint32_t ct_in = get_arg_val<uint32_t>(10);
    const uint32_t last_rows = get_arg_val<uint32_t>(11);
    const uint32_t last_lanes = get_arg_val<uint32_t>(12);
    const uint32_t slots = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr auto base_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    if (unit_begin >= unit_end) {
        return;
    }
#if defined(ARCH_BLACKHOLE)
    // Firmware leaves this RISC's L0 data cache (cfg0 bit 3, Blackhole only) off unless asked; the
    // gathers read each 16-byte line several times. Every read of NoC-written data follows a read
    // barrier, whose fence flushes the cache, and stores flush the lines they hit. The bit is
    // restored before returning.
    uint32_t cfg0;
    asm volatile("csrrci %0, 0x7c0, 0x8" : "=r"(cfg0)::"memory");
#endif
    const auto base = TensorAccessor(base_args, base_addr, TILE_BYTES);
    const auto out = TensorAccessor(out_args, out_addr, TILE_BYTES);
    const auto table_src = TensorAccessor(table_args, table_addr, TABLE_PAGE_BYTES);

    const uint32_t table_l1 = get_write_ptr(cb);
    const uint32_t buffers[2] = {table_l1 + table_bytes, table_l1 + table_bytes + slots * TILE_BYTES};
    const uint32_t gathered = buffers[1] + slots * TILE_BYTES;
    for (uint32_t page = 0; page < table_bytes / TABLE_PAGE_BYTES; page++) {
        noc_async_read_page(page, table_src, table_l1 + page * TABLE_PAGE_BYTES);
    }
    for (uint32_t b = 0; b < 2; b++) {
        tt_l1_ptr uint32_t* zero = reinterpret_cast<tt_l1_ptr uint32_t*>(buffers[b] + (slots - 1) * TILE_BYTES);
        for (uint32_t i = 0; i < TILE_BYTES / 4; i++) {
            zero[i] = 0;
        }
    }
    noc_async_read_barrier();
    const tt_l1_ptr uint32_t* table = reinterpret_cast<const tt_l1_ptr uint32_t*>(table_l1);

    const uint32_t units_per_chunk = nblk * rt_out;
    auto read_unit = [&](uint32_t unit, uint32_t buffer) {
        const tt_l1_ptr uint32_t* chunk = table + table[1 + unit / units_per_chunk];
        const uint32_t rem = unit % units_per_chunk;
        const uint32_t row_tile = (chunk[0] + rem / rt_out) * rt_in + chunk[1] + rem % rt_out;
        const uint32_t n_src = chunk[2];
        for (uint32_t i = 0; i < n_src; i++) {
            noc_async_read_page(row_tile * ct_in + chunk[5 + i], base, buffer + i * TILE_BYTES);
        }
    };

    uint32_t ring = 0;
    read_unit(unit_begin, buffers[0]);
    for (uint32_t unit = unit_begin, which = 0; unit < unit_end; unit++, which ^= 1) {
        const uint32_t buffer = buffers[which];
        noc_async_read_barrier();
        if (unit + 1 < unit_end) {
            // The other buffer's slots may still be leaving through verbatim copies
            noc_async_writes_flushed();
            read_unit(unit + 1, buffers[which ^ 1]);
        }

        const tt_l1_ptr uint32_t* chunk = table + table[1 + unit / units_per_chunk];
        const uint32_t rem = unit % units_per_chunk;
        const uint32_t n = rem / rt_out;
        const uint32_t rt = rem % rt_out;
        const uint32_t n_src = chunk[2];
        const uint32_t flags = chunk[3];
        const uint32_t n_rec = chunk[4];
        const tt_l1_ptr uint32_t* src = chunk + 5;
        const tt_l1_ptr uint32_t* rec = src + n_src;

        bool zeroed = false;
        if (rt + 1 == rt_out && last_rows < 32) {
            for (uint32_t i = 0; i < n_src; i++) {
                zero_rows(buffer + i * TILE_BYTES, last_rows);
            }
            zeroed = true;
        }
        if (flags & ZERO_PAD_LANES) {
            for (uint32_t i = 0; i < n_src; i++) {
                if (src[i] + 1 == ct_in) {
                    zero_lanes(buffer + i * TILE_BYTES, last_lanes);
                    zeroed = true;
                }
            }
        }
        if (zeroed) {
            drain(buffer);
        }

        const uint32_t out_row = n * rt_out + rt;
        const tt_l1_ptr uint16_t* halves = reinterpret_cast<const tt_l1_ptr uint16_t*>(buffer);
        // A gather already built into the other gathered tile by the record before it
        bool built = false;
        for (uint32_t r = 0; r < n_rec; r++) {
            const uint32_t head = rec[0];
            const uint32_t n_dst = (head >> 16) & 0x7fff;
            const uint32_t t = head & 0xffff;
            const tt_l1_ptr uint32_t* dst = rec + 1;
            uint32_t tile;
            if (head & GATHER) {
                const tt_l1_ptr uint32_t* pairs = dst + n_dst;
                tile = gathered + ring * TILE_BYTES;
                ring ^= 1;
                if (!built) {
                    // The previous writes out of both gathered tiles have left L1
                    noc_async_writes_flushed();
                    const tt_l1_ptr uint32_t* next = pairs + 16;
                    bool pair = r + 1 < n_rec && (next[0] & GATHER);
                    const tt_l1_ptr uint32_t* next_pairs = pair ? next + 1 + ((next[0] >> 16) & 0x7fff) : next;
                    for (uint32_t k = 0; pair && k < 16; k++) {
                        pair = next_pairs[k] == pairs[k] + 0x10001;
                    }
                    tt_l1_ptr uint32_t* words = reinterpret_cast<tt_l1_ptr uint32_t*>(tile);
                    tt_l1_ptr uint32_t* other = reinterpret_cast<tt_l1_ptr uint32_t*>(gathered + ring * TILE_BYTES);
                    for (uint32_t g = 0; g < 4; g++) {
                        const uint32_t at = (g >> 1) * 128 + (g & 1) * 4;
                        if (pair) {
                            gather_group_pair(words + at, other + at, halves, pairs + 4 * g);
                        } else {
                            gather_group(words + at, halves, pairs + 4 * g);
                        }
                    }
                    drain(tile);
                    built = pair;
                } else {
                    built = false;
                }
                rec = pairs + 16;
            } else {
                tile = buffer + dst[n_dst] * TILE_BYTES;
                rec = dst + n_dst + 1;
            }
            for (uint32_t d = 0; d < n_dst; d++) {
                noc_async_write_page((dst[d] * units_per_chunk + out_row) * ct_out + t, out, tile);
            }
        }
    }
    noc_async_write_barrier();
#if defined(ARCH_BLACKHOLE)
    asm volatile("fence\n csrs 0x7c0, %0" ::"r"(cfg0 & 0x8) : "memory");
#endif
}

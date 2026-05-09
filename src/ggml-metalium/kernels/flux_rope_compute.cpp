// Flux-RoPE compute kernel (interleaved mode)
//
// PE format: [L, D] in TILE, pe[l, 2*p] = cos, pe[l, 2*p+1] = -sin
// Rotation: x' = x*cos + y*sin, y' = -x*sin + y*cos
//
// Dest layout: slot 0 (regs 0-31) = src, slot 1 (regs 32-63) = PE
//
// WORKAROUND: The TT SFPI compiler (rvtt_synth_renumber pass) crashes with an ICE
// when cross-multiplying 4 vFloat operands from two different dst_reg tile slots
// in a single expression (e.g. `x*c - y*ns` where x,y are from slot 0 and c,ns
// from slot 1). To avoid this, the rotation is split into separate multiply and
// accumulate steps so no expression mixes more than 2 cross-slot operands.
//
// Without this compiler bug, the rotation would be written as:
//
//   vFloat c  = dst_reg[p + i * 2];       // cos from PE
//   vFloat ns = dst_reg[p + i * 2 + 1];   // -sin from PE
//   vFloat x  = dst_reg[s + i * 2];       // x_even from src
//   vFloat y  = dst_reg[s + i * 2 + 1];   // x_odd from src
//   dst_reg[s + i * 2]     = x * c - y * ns;   // x' = x*cos + y*sin
//   dst_reg[s + i * 2 + 1] = x * ns + y * c;   // y' = -x*sin + y*cos

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/identity.h"

// #include <tools/profiler/kernel_profiler.hpp>

#ifdef TRISC_MATH
using namespace sfpi;

inline void flux_rope_tile()
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    for (int face = 0; face < 4; face++) {
        int s = face * 8;
        int p = 32 + face * 8;

        for (int i = 0; i < 4; i++) {
            // Read originals before any overwrites
            vFloat x = dst_reg[s + i * 2];
            vFloat y = dst_reg[s + i * 2 + 1];

            // Even output: x' = x*cos - y*(-sin) = x*cos + y*sin
            // Step 1: tmp = x * cos  (1 cross-slot multiply)
            vFloat tmp = x * dst_reg[p + i * 2];
            // Step 2: tmp -= y * (-sin)  (1 cross-slot multiply, separate expression)
            tmp -= y * dst_reg[p + i * 2 + 1];
            dst_reg[s + i * 2] = tmp;

            // Odd output: y' = x*(-sin) + y*cos
            // Step 1: tmp2 = x * (-sin)
            vFloat tmp2 = x * dst_reg[p + i * 2 + 1];
            // Step 2: tmp2 += y * cos
            tmp2 += y * dst_reg[p + i * 2];
            dst_reg[s + i * 2 + 1] = tmp2;
        }
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    TTI_SETC16(2, 0);
}
#endif

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_src = tt::CBIndex::c_0;
    constexpr uint32_t cb_pe  = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_src, cb_out);
    pack_reconfig_data_format(cb_out);

    for (uint32_t t = 0; t < n_tiles; t++) {
        cb_wait_front(cb_src, 1);
        cb_wait_front(cb_pe, 1);

        tile_regs_acquire();

        copy_tile_init(cb_src);
        copy_tile(cb_src, 0, 0);

        copy_tile_init(cb_pe);
        copy_tile(cb_pe, 0, 1);

        MATH(flux_rope_tile());

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 2);
        pack_tile(0, cb_out, 0);
        tile_regs_release();
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_src, 1);
        cb_pop_front(cb_pe, 1);
    }
}

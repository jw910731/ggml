#include <cstdint>

void kernel_main() {
    uint32_t src_addr            = get_arg_val<uint32_t>(0);
    uint32_t pe_addr             = get_arg_val<uint32_t>(1);
    uint32_t tile_begin          = get_arg_val<uint32_t>(2);
    uint32_t tile_end            = get_arg_val<uint32_t>(3);
    uint32_t pe_tiles_per_batch  = get_arg_val<uint32_t>(4);  // Lt * Dt

    constexpr uint32_t cb_src = tt::CBIndex::c_0;
    constexpr uint32_t cb_pe  = tt::CBIndex::c_1;

    const uint32_t src_tile_size = get_tile_size(cb_src);
    const uint32_t pe_tile_size = get_tile_size(cb_pe);

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, src_tile_size);

    constexpr auto pe_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto pe = TensorAccessor(pe_args, pe_addr, pe_tile_size);

    for (uint32_t tile_id = tile_begin; tile_id < tile_end; tile_id++) {
        // tile_id indexes into the src tensor [B, L, D] in tile-linearized order:
        //   tile_id = b * pe_tiles_per_batch + lt * Dt + dt
        // PE is [L, D], so PE tile index = tile_id % pe_tiles_per_batch
        uint32_t pe_tile_id = tile_id % pe_tiles_per_batch;

        // Read src tile
        cb_reserve_back(cb_src, 1);
        uint32_t cb_src_wr = get_write_ptr(cb_src);
        noc_async_read_tile(tile_id, src, cb_src_wr);

        // Read corresponding PE tile (broadcast over batch dimension)
        cb_reserve_back(cb_pe, 1);
        uint32_t cb_pe_wr = get_write_ptr(cb_pe);
        noc_async_read_tile(pe_tile_id, pe, cb_pe_wr);

        noc_async_read_barrier();
        cb_push_back(cb_src, 1);
        cb_push_back(cb_pe, 1);
    }
}
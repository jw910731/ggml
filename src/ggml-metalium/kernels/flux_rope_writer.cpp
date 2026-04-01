#include <cstdint>

void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    uint32_t tile_begin = get_arg_val<uint32_t>(1);
    uint32_t tile_end   = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(dst_args, dst_addr, tile_size_bytes);

    for (uint32_t tile_id = tile_begin; tile_id < tile_end; tile_id++) {
        cb_wait_front(cb_out, 1);
        uint32_t cb_out_addr = get_read_ptr(cb_out);
        noc_async_write_tile(tile_id, dst, cb_out_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
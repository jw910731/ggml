#include "flux_rope.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include <ttnn/tensor/layout/layout.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "utils.hpp"

using namespace tt::tt_metal;

namespace ttggml {
using namespace ttnn;

// ---- FluxRoPEDeviceOperation ----

flux_rope_device::FluxRoPEDeviceOperation::program_factory_t
flux_rope_device::FluxRoPEDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& /*tensor_args*/)
{
    return program::FluxRoPEProgramFactory{};
}

flux_rope_device::FluxRoPEDeviceOperation::spec_return_value_t
flux_rope_device::FluxRoPEDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args)
{
    const auto& src = tensor_args.src;
    return TensorSpec(
        src.logical_shape(),
        tt::tt_metal::TensorLayout(
            src.dtype(),
            tt::tt_metal::PageConfig(src.layout()),
            operation_attributes.output_mem_config));
}

flux_rope_device::FluxRoPEDeviceOperation::tensor_return_value_t
flux_rope_device::FluxRoPEDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args)
{
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.src.device());
}

void flux_rope_device::FluxRoPEDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args)
{
    const auto& src = tensor_args.src;
    const auto& pe = tensor_args.pe;

    TT_FATAL(src.layout() == tt::tt_metal::Layout::TILE, "Source tensor must be TILE layout");
    TT_FATAL(pe.layout() == tt::tt_metal::Layout::TILE, "PE tensor must be TILE layout");
    TT_FATAL(src.storage_type() == ttnn::StorageType::DEVICE, "Source must be on device");
    TT_FATAL(pe.storage_type() == ttnn::StorageType::DEVICE, "PE must be on device");

    const auto& src_shape = src.logical_shape();
    const auto& pe_shape = pe.logical_shape();

    // Metalium backend always pads to 4D with leading 1s:
    //   src: GGML [D, L, B] → TT [1, B, L, D]
    //   pe:  GGML [D, L]    → TT [1, 1, L, D]
    // Use [-N] indexing to access dimensions from the right.
    TT_FATAL(src_shape.size() >= 2 && src_shape.size() <= 4,
        "Source must be 2D-4D, got {}D shape {}", src_shape.size(), src_shape);
    TT_FATAL(pe_shape.size() >= 2 && pe_shape.size() <= 4,
        "PE must be 2D-4D, got {}D shape {}", pe_shape.size(), pe_shape);

    // L and D must match between src and pe
    TT_FATAL(src_shape[-2] == pe_shape[-2],
        "L dimension mismatch: src L={}, pe L={}", src_shape[-2], pe_shape[-2]);
    TT_FATAL(src_shape[-1] == pe_shape[-1],
        "D dimension mismatch: src D={}, pe D={}", src_shape[-1], pe_shape[-1]);

    if (operation_attributes.rope_interleaved) {
        TT_FATAL(src_shape[-1] % 2 == 0, "D must be even for interleaved RoPE, got D={}", src_shape[-1]);
    } else {
        TT_FATAL(false, "Non-interleaved Flux-RoPE not yet implemented");
    }
}

void flux_rope_device::FluxRoPEDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args)
{
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

std::tuple<flux_rope_device::operation_attributes_t, flux_rope_device::tensor_args_t>
flux_rope_device::FluxRoPEDeviceOperation::invoke(
    const Tensor& src, const Tensor& pe, bool rope_interleaved)
{
    return {
        operation_attributes_t{src.memory_config(), rope_interleaved},
        tensor_args_t{src, pe}
    };
}

// ---- FluxRoPEProgramFactory ----

flux_rope_device::program::FluxRoPEProgramFactory::cached_program_t
flux_rope_device::program::FluxRoPEProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value)
{
    Program program{};

    const auto& src_tensor = tensor_args.src;
    const auto& pe_tensor = tensor_args.pe;
    const auto& output_tensor = tensor_return_value;

    IDevice* device = src_tensor.device();
    auto* src = src_tensor.buffer();
    auto* pe_buf = pe_tensor.buffer();
    auto* dst = output_tensor.buffer();

    // src shape: [B, L, D]
    // In TT format (outer-to-inner): B is batch, L and D are tiled dims
    const auto& src_shape = src_tensor.padded_shape();
    const uint32_t B = src_shape[-3];
    const uint32_t L_padded = src_shape[-2];
    const uint32_t D_padded = src_shape[-1];

    const uint32_t Lt = L_padded / 32;  // tiles in L dimension (already padded to tile boundary)
    const uint32_t Dt = D_padded / 32;  // tiles in D dimension
    const uint32_t total_src_tiles = B * Lt * Dt;
    const uint32_t pe_tiles_per_batch = Lt * Dt;  // PE is [L, D], broadcast over B

    auto core_grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2,
          work_per_core1, work_per_core2] =
        split_work_to_cores(core_grid, total_src_tiles);

    // Circular buffers
    // CB0: src input tile (double-buffered for pipelining with reader)
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_0, 2, src_tensor.dtype());
    // CB1: PE input tile (double-buffered for pipelining with reader)
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_1, 2, pe_tensor.dtype());
    // CB_out: output tile (size 4 to satisfy cb_reserve_back(2) + packer alignment)
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_16, 4, output_tensor.dtype());

    // Reader kernel compile-time args: TensorAccessor params for src and pe
    std::vector<uint32_t> reader_compile_args;
    TensorAccessorArgs(*src).append_to(reader_compile_args);
    TensorAccessorArgs(*pe_buf).append_to(reader_compile_args);

    KernelHandle reader = CreateMetaliumKernel(program, "flux_rope_reader", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_args,
        .defines = {},
        .named_compile_args = {}
    });

    // Writer kernel compile-time args
    std::vector<uint32_t> writer_compile_args;
    TensorAccessorArgs(*dst).append_to(writer_compile_args);

    KernelHandle writer = CreateMetaliumKernel(program, "flux_rope_writer", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_args,
        .defines = {},
        .named_compile_args = {}
    });

    // Compute kernel
    KernelHandle compute = CreateMetaliumKernel(program, "flux_rope_compute", all_cores, ComputeConfig{
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = {},
        .compile_args = {},
        .defines = {},
        .named_compile_args = {},
    });

    // Set runtime args per core
    uint32_t tile_offset = 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            uint32_t n_tiles = 0;
            if (core_group_1.contains(core)) {
                n_tiles = work_per_core1;
            } else if (core_group_2.contains(core)) {
                n_tiles = work_per_core2;
            }

            // Reader runtime args:
            //   src_addr, pe_addr, tile_begin, tile_end, pe_tiles_per_batch
            SetRuntimeArgs(program, reader, core, std::vector<uint32_t>{
                src->address(),
                pe_buf->address(),
                tile_offset,
                tile_offset + n_tiles,
                pe_tiles_per_batch
            });

            // Compute runtime args:
            //   n_tiles
            SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{
                n_tiles
            });

            // Writer runtime args:
            //   dst_addr, tile_begin, tile_end
            SetRuntimeArgs(program, writer, core, std::vector<uint32_t>{
                dst->address(),
                tile_offset,
                tile_offset + n_tiles
            });

            tile_offset += n_tiles;
        }
    }

    return {std::move(program), shared_variables_t{reader, writer, compute, all_cores}};
}

void flux_rope_device::program::FluxRoPEProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value)
{
    auto& program = cached_program.program;
    const auto& reader = cached_program.shared_variables.reader;
    const auto& writer = cached_program.shared_variables.writer;
    const auto& all_cores = cached_program.shared_variables.all_cores;

    auto* src_buffer = tensor_args.src.buffer();
    auto* pe_buffer = tensor_args.pe.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            {
                auto& runtime_args = GetRuntimeArgs(program, reader, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = pe_buffer->address();
            }
            {
                auto& runtime_args = GetRuntimeArgs(program, writer, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    }
}

// ---- prim::flux_rope ----

ttnn::Tensor prim::flux_rope(const Tensor& src, const Tensor& pe, bool rope_interleaved) {
    auto [attrs, args] = flux_rope_device::FluxRoPEDeviceOperation::invoke(src, pe, rope_interleaved);
    return ttnn::device_operation::detail::launch<flux_rope_device::FluxRoPEDeviceOperation>(attrs, args);
}

// ---- flux_rope ----

ttnn::Tensor flux_rope(
    const Tensor& src_tensor, const Tensor& pe_tensor, bool rope_interleaved)
{
    return ttggml::prim::flux_rope(src_tensor, pe_tensor, rope_interleaved);
}

} // namespace ttggml
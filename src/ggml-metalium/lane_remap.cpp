#include "lane_remap.hpp"
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

namespace {

constexpr uint32_t TILE_BYTES = 2048;

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

} // namespace

// ---- LaneRemapDeviceOperation ----

lane_remap_device::LaneRemapDeviceOperation::program_factory_t
lane_remap_device::LaneRemapDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/)
{
    return program::LaneRemapProgramFactory{};
}

lane_remap_device::LaneRemapDeviceOperation::spec_return_value_t
lane_remap_device::LaneRemapDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/)
{
    const auto& s = operation_attributes.out_shape;
    return TensorSpec(
        ttnn::Shape({s[0], s[1], s[2], s[3]}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.output_mem_config));
}

lane_remap_device::LaneRemapDeviceOperation::tensor_return_value_t
lane_remap_device::LaneRemapDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args)
{
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.base.device());
}

void lane_remap_device::LaneRemapDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args)
{
    const auto& base = tensor_args.base;
    const auto& table = tensor_args.table;
    TT_FATAL(base.storage_type() == ttnn::StorageType::DEVICE && table.storage_type() == ttnn::StorageType::DEVICE,
        "lane_remap operands must be on device");
    TT_FATAL(base.layout() == Layout::TILE && base.dtype() == DataType::BFLOAT16, "lane_remap base must be bf16 TILE");
    TT_FATAL(!base.memory_config().is_sharded() && !table.memory_config().is_sharded(),
        "lane_remap operands must be interleaved");
    TT_FATAL(table.dtype() == DataType::UINT32 && table.layout() == Layout::ROW_MAJOR &&
        table.logical_shape().volume() == attrs.table_words && table.logical_shape()[-1] == ML_REMAP_PAGE_WORDS,
        "lane_remap table must be uint32 rows of {} words", ML_REMAP_PAGE_WORDS);

    const auto& logical = base.logical_shape();
    const auto& padded = base.padded_shape();
    TT_FATAL(logical.rank() >= 2, "lane_remap base must be at least 2D");
    uint64_t blocks = 1;
    for (int i = 0; i + 2 < (int)padded.rank(); i++) {
        blocks *= padded[i];
    }
    TT_FATAL(blocks == attrs.in_blocks && logical[-2] == attrs.in_rows && logical[-1] == attrs.in_lanes,
        "lane_remap base {} does not match the plan's grid {}x{}x{}", logical, attrs.in_blocks, attrs.in_rows,
        attrs.in_lanes);
}

void lane_remap_device::LaneRemapDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args)
{
    validate_on_program_cache_miss(attrs, tensor_args);
}

ttsl::hash::hash_t lane_remap_device::LaneRemapDeviceOperation::compute_program_hash(
    const operation_attributes_t& a, const tensor_args_t& tensor_args)
{
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<LaneRemapDeviceOperation>, a.out_shape, a.in_blocks, a.in_rows, a.in_lanes, a.nblk,
        a.nrow, a.n_chunks, a.slots, a.table_words, a.table_hash, a.output_mem_config,
        tensor_args.base.tensor_spec(), tensor_args.table.tensor_spec());
}

std::tuple<lane_remap_device::operation_attributes_t, lane_remap_device::tensor_args_t>
lane_remap_device::LaneRemapDeviceOperation::invoke(
    const Tensor& base, const ml_plan& plan, const ml_remap& remap, const Tensor& table)
{
    const auto u32 = [](int64_t v) { return (uint32_t)v; };
    return {
        operation_attributes_t{
            .out_shape = {u32(plan.out_shape[0]), u32(plan.out_shape[1]), u32(plan.out_shape[2]),
                          u32(plan.out_shape[3])},
            .in_blocks = u32(plan.in.blocks),
            .in_rows = u32(plan.in.R),
            .in_lanes = u32(plan.in.W),
            .nblk = u32(remap.nblk),
            .nrow = u32(remap.nrow),
            .n_chunks = u32(remap.n_chunks),
            .slots = u32(remap.slots),
            .table_words = u32(remap.table.size()),
            .table_hash = remap.hash,
            .output_mem_config = base.memory_config(),
        },
        tensor_args_t{base, table}};
}

// ---- LaneRemapProgramFactory ----

lane_remap_device::program::LaneRemapProgramFactory::cached_program_t
lane_remap_device::program::LaneRemapProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output)
{
    Program program{};

    const auto& base = tensor_args.base;
    auto* base_buf = base.buffer();
    auto* table_buf = tensor_args.table.buffer();
    auto* out_buf = output.buffer();
    IDevice* device = base.device();

    const auto& padded = base.padded_shape();
    const uint32_t rt_in = padded[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t ct_in = padded[-1] / tt::constants::TILE_WIDTH;
    const uint32_t rt_out = div_up(attrs.nrow, tt::constants::TILE_HEIGHT);
    const uint32_t ct_out = div_up(attrs.out_shape[3], tt::constants::TILE_WIDTH);
    const uint32_t last_rows = attrs.nrow - (rt_out - 1) * tt::constants::TILE_HEIGHT;
    const uint32_t last_lanes = attrs.in_lanes - (ct_in - 1) * tt::constants::TILE_WIDTH;
    const uint32_t n_units = attrs.n_chunks * attrs.nblk * rt_out;
    const uint32_t table_bytes = attrs.table_words * sizeof(uint32_t);

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(device->compute_with_storage_grid_size(), n_units);

    // Each RISC works in L1 of its own: the table, two slot buffers it alternates between (one
    // being read while the other is processed) and two tiles to gather into
    const uint32_t scratch = table_bytes + (2 * attrs.slots + 2) * TILE_BYTES;
    const tt::CBIndex cbs[2] = {tt::CBIndex::c_0, tt::CBIndex::c_1};
    shared_variables_t shared{};
    for (int risc = 0; risc < 2; risc++) {
        MakeCircularBuffer(program, all_cores, cbs[risc], scratch, scratch, tt::DataFormat::Float16_b);
        std::vector<uint32_t> compile_args = {(uint32_t)cbs[risc]};
        TensorAccessorArgs(*base_buf).append_to(compile_args);
        TensorAccessorArgs(*out_buf).append_to(compile_args);
        TensorAccessorArgs(*table_buf).append_to(compile_args);
        shared.movers[risc] = CreateMetaliumKernel(program, "lane_remap_mover", all_cores, DataMovementConfig{
            .processor = risc == 0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1,
            .noc = risc == 0 ? NOC::RISCV_0_default : NOC::RISCV_1_default,
            .compile_args = compile_args,
            .defines = {},
            .named_compile_args = {},
        });
    }

    uint32_t unit = 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            const uint32_t n = core_group_1.contains(core) ? units_per_core_1 :
                               core_group_2.contains(core) ? units_per_core_2 : 0;
            const uint32_t mid = unit + (n + 1) / 2;
            const uint32_t spans[2][2] = {{unit, mid}, {mid, unit + n}};
            for (int risc = 0; risc < 2; risc++) {
                SetRuntimeArgs(program, shared.movers[risc], core, std::vector<uint32_t>{
                    base_buf->address(),
                    out_buf->address(),
                    table_buf->address(),
                    table_bytes,
                    spans[risc][0],
                    spans[risc][1],
                    attrs.nblk,
                    rt_out,
                    ct_out,
                    rt_in,
                    ct_in,
                    last_rows,
                    last_lanes,
                    attrs.slots,
                });
            }
            shared.cores.push_back(core);
            unit += n;
        }
    }
    TT_FATAL(unit == n_units, "lane_remap assigned {} of {} units", unit, n_units);

    return {std::move(program), std::move(shared)};
}

void lane_remap_device::program::LaneRemapProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output)
{
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    for (const auto& core : shared.cores) {
        for (auto mover : shared.movers) {
            auto& args = GetRuntimeArgs(program, mover, core);
            args[0] = tensor_args.base.buffer()->address();
            args[1] = output.buffer()->address();
            args[2] = tensor_args.table.buffer()->address();
        }
    }
}

// ---- lane_remap ----

ttnn::Tensor lane_remap_table(const ml_remap& remap, ttnn::MeshDevice* device)
{
    const TensorSpec spec(
        ttnn::Shape({1, 1, (uint32_t)(remap.table.size() / ML_REMAP_PAGE_WORDS), (uint32_t)ML_REMAP_PAGE_WORDS}),
        tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    return Tensor::from_vector(remap.table, spec, device);
}

ttnn::Tensor lane_remap(const Tensor& base, const ml_plan& plan, const ml_remap& remap, const Tensor& table)
{
    auto [attrs, args] = lane_remap_device::LaneRemapDeviceOperation::invoke(base, plan, remap, table);
    return ttnn::device_operation::detail::launch<lane_remap_device::LaneRemapDeviceOperation>(attrs, args);
}

} // namespace ttggml

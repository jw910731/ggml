#pragma once

#include <array>
#include <cstdint>
#include <tuple>
#include <variant>

#include <ttnn/device_operation.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "metalium_layout.hpp"

// lane_remap: a row-local plan (metalium_layout.hpp) as one pure data-movement program. Both
// data-movement RISCs of every core read base tiles, move halfwords between tiles in L1 and write
// the result, so every bit pattern, NaN payloads, -0 and subnormals included, arrives unchanged.
// Pad rows and pad lanes of the result are zero.

namespace ttggml {
using namespace ttnn;

namespace lane_remap_device {

struct operation_attributes_t {
    std::array<uint32_t, 4> out_shape;  // TT order
    uint32_t in_blocks;
    uint32_t in_rows;
    uint32_t in_lanes;
    uint32_t nblk;
    uint32_t nrow;
    uint32_t n_chunks;
    uint32_t slots;
    uint32_t table_words;
    uint64_t table_hash;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    const Tensor& base;   // bf16 TILE, interleaved
    const Tensor& table;  // uint32 [1, 1, pages, ML_REMAP_PAGE_WORDS] ROW_MAJOR, interleaved
};

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {

struct LaneRemapProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle movers[2];
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

} // namespace program

struct LaneRemapDeviceOperation {
    using operation_attributes_t = lane_remap_device::operation_attributes_t;
    using tensor_args_t = lane_remap_device::tensor_args_t;
    using spec_return_value_t = lane_remap_device::spec_return_value_t;
    using tensor_return_value_t = lane_remap_device::tensor_return_value_t;
    using program_factory_t = std::variant<program::LaneRemapProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& base, const ml_plan& plan, const ml_remap& remap, const Tensor& table);
};

} // namespace lane_remap_device

// The table of `remap` (ml_build_remap of `plan`) as a device tensor
ttnn::Tensor lane_remap_table(const ml_remap& remap, ttnn::MeshDevice* device);

// `plan` applied to `base` (whose grid is plan.in), returned in the TT shape plan.out_shape
ttnn::Tensor lane_remap(const Tensor& base, const ml_plan& plan, const ml_remap& remap, const Tensor& table);

} // namespace ttggml

#pragma once

#include <tuple>
#include <variant>

#include <ttnn/device_operation.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace ttggml {
using namespace ttnn;

namespace flux_rope_device {

struct operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
    bool rope_interleaved;
};

struct tensor_args_t {
    const Tensor& src;   // [B, L, D] in TILE, where B = N * n_head
    const Tensor& pe;    // [L, D] in TILE, preprocessed: even cols = cos, odd cols = -sin
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {

struct FluxRoPEProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader;
        tt::tt_metal::KernelHandle writer;
        tt::tt_metal::KernelHandle compute;
        tt::tt_metal::CoreRangeSet all_cores;
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

struct FluxRoPEDeviceOperation {
    using operation_attributes_t = flux_rope_device::operation_attributes_t;
    using tensor_args_t = flux_rope_device::tensor_args_t;
    using spec_return_value_t = flux_rope_device::spec_return_value_t;
    using tensor_return_value_t = flux_rope_device::tensor_return_value_t;
    using program_factory_t = std::variant<program::FluxRoPEProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&,
        const tensor_args_t&);

    static void validate_on_program_cache_hit(
        const operation_attributes_t&,
        const tensor_args_t&);

    static void validate_on_program_cache_miss(
        const operation_attributes_t&,
        const tensor_args_t&);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&,
        const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&,
        const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& src,
        const Tensor& pe,
        bool rope_interleaved);
};

} // namespace flux_rope_device

namespace prim {
ttnn::Tensor flux_rope(const Tensor& src, const Tensor& pe, bool rope_interleaved);
} // namespace prim

ttnn::Tensor flux_rope(
    const Tensor& src_tensor,
    const Tensor& pe_tensor,
    bool rope_interleaved = true);

} // namespace ttggml

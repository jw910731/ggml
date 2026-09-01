if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()

message(STATUS "Metalium found.")

ggml_add_backend_library(ggml-metalium
    ggml-metalium.cpp
    utils.cpp
    tmp_rope.cpp
    tmp_mul_mat.cpp
    tmp_soft_max.cpp
    flux_rope.cpp
)

target_include_directories(ggml-metalium PRIVATE
    $ENV{TT_METAL_HOME}/build/include
    $ENV{TT_METAL_HOME}/build/include/metalium-thirdparty
    # Metalium
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/api
    $ENV{TT_METAL_HOME}/tt_stl
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api/umd/device
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public/tracy
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public/
    $ENV{TT_METAL_HOME}/tt_metal/hostdevcommon/api/
    $ENV{TT_METAL_HOME}/tt_metal/include

    # TTNN
    $ENV{TT_METAL_HOME}/ttnn
    $ENV{TT_METAL_HOME}/ttnn/api
    $ENV{TT_METAL_HOME}/ttnn/cpp

    # Parent GGML stuff
    ..
)

target_link_directories(ggml-metalium PRIVATE
    $ENV{TT_METAL_HOME}/build/ttnn
    $ENV{TT_METAL_HOME}/build/tt_metal
)

target_link_libraries(ggml-metalium PRIVATE
    # Metalium
    tt_metal

    # TTNN
    $ENV{TT_METAL_HOME}/build/ttnn/_ttnncpp.so

    ggml-base
)

if(GGML_METALIUM_EMBED_KERNELS)
    add_library(ggml-metalium-kernels STATIC)
    set(GGML_METALIUM_EMBED_KERNELS
            rope_neox_reader
            rope_neox_compute
            rope_neox_writer
            rope_normal_reader
            rope_normal_compute
            rope_normal_writer
            mul_mat_reader
            mul_mat_compute
            mul_mat_writer
            soft_max_reader
            soft_max_writer
            soft_max_compute
            flux_rope_reader
            flux_rope_compute
            flux_rope_writer
    )
    include(cmake/add_kernel.cmake)
    foreach(kernel IN ITEMS ${GGML_METALIUM_EMBED_KERNELS})
        add_kernel(ggml-metalium-kernels ${kernel})
    endforeach()
    register_kernels(ggml-metalium "${GGML_METALIUM_EMBED_KERNELS}")
    target_link_libraries(ggml-metalium PRIVATE ggml-metalium-kernels)
    target_compile_definitions(ggml-metalium PRIVATE GGML_METALIUM_EMBED_KERNELS)
    set_target_properties(ggml-metalium-kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(ggml-metalium-kernels PRIVATE -Wno-missing-declarations)
endif()


# ttPseudoRowMajor (ttprm) -- fused, view-aware TTNN ops. Opt-in.
#
# We deliberately do NOT add_subdirectory() the submodule: its own CMakeLists
# declares `ttmetal`/`ttnnlib` IMPORTED INTERFACE targets and pulls fmt/yaml-cpp/
# Boost/CLI11 through CPM, all of which overlap-but-differ from the include/link
# set this backend already carries. Instead we compile the single TU that links
# TTNN (view_realize_op.cpp) straight into ggml-metalium and reuse our own flags.
if(GGML_METALIUM_TTPRM)
    set(TTPRM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/ttPseudoRowMajor/ttprm")
    if(NOT EXISTS "${TTPRM_DIR}/view_realize_op.cpp")
        message(FATAL_ERROR
            "GGML_METALIUM_TTPRM=ON but ${TTPRM_DIR} is empty. "
            "Run: git submodule update --init --recursive")
    endif()
    message(STATUS "Metalium: ttprm fused view ops enabled (${TTPRM_DIR})")

    target_sources(ggml-metalium PRIVATE
        "${TTPRM_DIR}/view_realize_op.cpp"
        ttprm_bridge.cpp
    )
    target_include_directories(ggml-metalium PRIVATE "${TTPRM_DIR}")
    # TTPRM_ROOT is a *runtime* dependency: the device kernels are JIT-compiled
    # from ${TTPRM_ROOT}/kernel/*.cpp, so the source tree must still be present
    # when sd-cli runs. Embedding them the way GGML_METALIUM_EMBED_KERNELS does
    # for this backend's own kernels would need ttprm-side changes.
    #
    # Those kernels are also not all current: against tt-metal v0.78.0-dev the realize
    # and eltwise kernels build, but view_norm_compute.cpp does not
    # (ckernel::math::set_addr_mod_base is gone). The bridge latches an entry point off
    # after its first device-side failure, so a stale kernel costs one failed JIT build
    # per process rather than one per node.
    target_compile_definitions(ggml-metalium PRIVATE
        GGML_METALIUM_TTPRM
        TTPRM_ROOT="${TTPRM_DIR}")
    # ttprm's TU has its own `using namespace` block; keep our PCH off it, and
    # force-include the shim that restores the TTNN spellings it was written for.
    set_source_files_properties("${TTPRM_DIR}/view_realize_op.cpp" PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
        COMPILE_OPTIONS "-include;${CMAKE_CURRENT_SOURCE_DIR}/ttprm_compat.hpp")
endif()

target_compile_definitions(
    ggml-metalium PRIVATE
    FMT_HEADER_ONLY
    SPDLOG_FMT_EXTERNAL
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsized-deallocation")
endif()

target_precompile_headers(ggml-metalium PRIVATE metalium-pch.hpp)
target_compile_features(ggml-metalium PRIVATE cxx_std_20)

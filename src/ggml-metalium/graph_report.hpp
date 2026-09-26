#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

// ttnn::CONFIG's reporting flags (enable_logging, enable_graph_report, report_name, ...) are only stored by C++ ttnn;
// ttnn's Python frontend is what acts on them. This mirrors that frontend, with a ggml node, upload, readback or copy
// standing in for one ttnn Python operation.

// Applies the memory-profile preset and TTNN_CONFIG_OVERRIDES to ttnn::CONFIG. Call once before the device opens.
void ggml_metalium_report_init(bool memory_profile);

// Writes out the capture in progress. For process exit, while GraphTracker's thread-locals are still alive.
void ggml_metalium_report_flush();

// Lives for one call into the backend. With enable_logging it synchronizes the device around every operation. With
// enable_graph_report, calls from all threads take turns and are recorded into one GraphProcessor capture, so its
// device operations come in the order the device profiler sees them: ttnn-visualizer links a memory report to a
// performance report of the same run only then. Captures go to report_path as graph_capture_g<N>.json for ttnn's
// graph_report.py to import.
class ggml_metalium_report {
public:
    explicit ggml_metalium_report(tt::tt_metal::distributed::MeshDevice & device);
    ~ggml_metalium_report();

    ggml_metalium_report(const ggml_metalium_report &) = delete;
    ggml_metalium_report & operator=(const ggml_metalium_report &) = delete;

    bool active() const { return logging || recording; }

    void op_begin(const std::string & name, const char * tensor_name,
                  std::vector<std::reference_wrapper<const ttnn::Tensor>> & inputs);
    void op_end(const ttnn::Tensor * output);

    // Writes the capture once this call leaves the backend
    void flush_on_exit() { flush = true; }

private:
    void synchronize();

    tt::tt_metal::distributed::MeshDevice & device;
    const bool logging;
    bool recording = false;
    bool flush = false;
    std::unique_lock<std::recursive_mutex> lock;
    std::string op_name;
    std::vector<std::uint64_t> input_ids;
    std::size_t op_depth = 0;
};

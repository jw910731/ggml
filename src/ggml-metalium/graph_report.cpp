#include "graph_report.hpp"

#include "ggml.h"

#include <algorithm>
#include <any>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <reflect>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <ttnn/config.hpp>
#include <ttnn/graph/graph_processor.hpp>
#include <ttnn/graph/graph_serialization.hpp>

namespace {

struct cb_allocation {
    tt::tt_metal::CoreRangeSet core_ranges;
    uint64_t address;
    uint64_t size;
};

// ttnn reports a program's circular buffers only when they are placed, on a program-cache miss, so every later launch
// of the program would show none. Keyed by the program's implementation, which the program cache keeps alive, and
// kept across captures for the same reason.
std::unordered_map<const tt::tt_metal::detail::ProgramImpl *, std::vector<cb_allocation>> known_programs;

struct buffer_record {
    uint32_t device_id;
    uint64_t address;
    uint64_t max_size_per_bank;
    int buffer_type;
    int buffer_layout;
    std::size_t allocation;  // ordinal among the capture's buffer_allocate events
};

// What graph_report.py makes of an L1 buffer's pages on one bank (buffer_chunks)
struct bank_chunk {
    uint32_t bank_id;
    tt::tt_metal::CoreCoord core;
    uint64_t start;
    uint64_t end;
    uint32_t pages;
};

struct l1_allocation {
    uint32_t device_id;
    uint64_t address;
    uint64_t page_size;
    int buffer_type;
    std::vector<bank_chunk> chunks;
};

// ttnn::reports::get_buffer_pages lists every page of every L1 buffer, which for sharded activations is millions of
// entries per allocation. Only the per-bank extent survives the import, so compute that for the one buffer.
std::vector<bank_chunk> bank_chunks(const tt::tt_metal::Buffer & buffer) {
    std::vector<bank_chunk> chunks;
    const auto * allocator = buffer.allocator();
    const uint64_t page_size = buffer.page_size();
    if (tt::tt_metal::is_sharded(buffer.buffer_layout())) {
        // get_buffer_page_mapping() computes the mapping once and caches it
        const auto & mapping = *const_cast<tt::tt_metal::Buffer &>(buffer).get_buffer_page_mapping();
        std::unordered_map<uint32_t, bank_chunk> by_core;
        for (const auto page : mapping) {
            const uint64_t address = buffer.address() + page.device_page * buffer.aligned_page_size();
            auto [it, inserted] = by_core.try_emplace(page.core_id, bank_chunk{0, {}, address, address, 0});
            it->second.start = std::min(it->second.start, address);
            it->second.end = std::max(it->second.end, address);
            ++it->second.pages;
        }
        for (auto & [core_id, chunk] : by_core) {
            chunk.core = mapping.all_cores[core_id];
            chunk.bank_id = allocator->get_bank_ids_from_logical_core(buffer.buffer_type(), chunk.core)[0];
            chunk.end += page_size;
            chunks.push_back(chunk);
        }
        return chunks;
    }
    // Interleaved page p lives on bank p % num_banks
    const uint32_t num_banks = allocator->get_num_banks(buffer.buffer_type());
    const uint32_t num_pages = buffer.num_pages();
    for (uint32_t bank = 0; bank < num_banks && bank < num_pages; ++bank) {
        const uint32_t pages = (num_pages - 1 - bank) / num_banks + 1;
        const uint64_t first = buffer.page_address(bank, bank);
        const uint64_t last = buffer.page_address(bank, bank + (pages - 1) * num_banks);
        chunks.push_back({bank, allocator->get_logical_core_from_bank_id(bank), std::min(first, last),
                          std::max(first, last) + page_size, pages});
    }
    return chunks;
}

nlohmann::json buffer_json(const buffer_record & b) {
    return {{"device_id", b.device_id},
            {"address", b.address},
            {"max_size_per_bank", b.max_size_per_bank},
            {"buffer_type", b.buffer_type},
            {"buffer_layout", b.buffer_layout}};
}

class ggml_metalium_graph_processor final : public ttnn::graph::GraphProcessor {
public:
    explicit ggml_metalium_graph_processor(tt::tt_metal::distributed::MeshDevice & mesh) :
        GraphProcessor(RunMode::NORMAL),
        mesh(mesh),
        chip(mesh.get_devices().size() == 1 ? mesh.get_devices()[0] : nullptr) {}

    void track_allocate(const tt::tt_metal::Buffer * buffer) override {
        replay_cbs();
        GraphProcessor::track_allocate(buffer);
        const std::size_t ordinal = n_allocations++;
        const buffer_record record{
            .device_id = static_cast<uint32_t>(buffer->device()->id()),
            .address = buffer->address(),
            .max_size_per_bank = buffer->aligned_size_per_bank(),
            .buffer_type = static_cast<int>(buffer->buffer_type()),
            .buffer_layout = static_cast<int>(buffer->buffer_layout()),
            .allocation = ordinal,
        };
        if (!buffer->is_l1()) {
            live_dram[record.address] = record;
            return;
        }
        live_l1[record.address] = record;
        l1_bytes += record.max_size_per_bank;
        undecided_chunks[ordinal] = {record.device_id, record.address, buffer->page_size(), record.buffer_type,
                                     bank_chunks(*buffer)};
        if (node && l1_bytes > node->peak_bytes) {
            node->peak_bytes = l1_bytes;
            node->peak.clear();
            for (const auto & [address, live] : live_l1) {
                node->peak.push_back(live);
            }
        }
    }

    void track_deallocate(tt::tt_metal::Buffer * buffer) override {
        replay_cbs();
        GraphProcessor::track_deallocate(buffer);
        if (!buffer->is_l1()) {
            live_dram.erase(buffer->address());
        } else if (auto it = live_l1.find(buffer->address()); it != live_l1.end()) {
            l1_bytes -= it->second.max_size_per_bank;
            live_l1.erase(it);
        }
    }

    void track_program(tt::tt_metal::Program * program, const tt::tt_metal::IDevice * device) override {
        replay_cbs();
        GraphProcessor::track_program(program, device);
        recording = nullptr;
        // With several chips, the circular buffers placed after a workload's programs cannot be told apart
        if (chip == nullptr) {
            return;
        }
        auto [known, first_launch] = known_programs.try_emplace(&program->impl());
        recording = &known->second;
        // A miss places the circular buffers right away and replaces what was known (the address may be a destroyed
        // program's); a cache hit places none, so the known ones are replayed at the launch's next event
        pending_replay = !first_launch;
    }

    void track_allocate_cb(const tt::tt_metal::CoreRangeSet & core_range_set, uint64_t addr, uint64_t size,
                           bool is_globally_allocated, const tt::tt_metal::IDevice * device) override {
        if (pending_replay && recording != nullptr) {
            pending_replay = false;
            recording->clear();
        }
        GraphProcessor::track_allocate_cb(core_range_set, addr, size, is_globally_allocated, on_mesh(device));
        // A globally allocated circular buffer lives in a tensor's buffer, whose address changes between launches
        if (recording != nullptr && !is_globally_allocated) {
            recording->push_back({core_range_set, addr, size});
        }
    }

    void track_allocate_dataflow_buffer(const tt::tt_metal::CoreRangeSet & core_range_set, uint64_t addr,
                                        uint64_t size, bool borrows_memory,
                                        const tt::tt_metal::IDevice * device) override {
        GraphProcessor::track_allocate_dataflow_buffer(core_range_set, addr, size, borrows_memory, on_mesh(device));
    }

    void track_allocate_scratchpad(const tt::tt_metal::CoreRangeSet & core_range_set, uint64_t addr, uint64_t size,
                                   const tt::tt_metal::IDevice * device) override {
        GraphProcessor::track_allocate_scratchpad(core_range_set, addr, size, on_mesh(device));
    }

    void track_deallocate_cb(const tt::tt_metal::IDevice * device) override {
        GraphProcessor::track_deallocate_cb(on_mesh(device));
    }

    void track_function_start(std::string_view function_name,
                              std::span<tt::tt_metal::TrackedArgument> input_parameters) override {
        replay_cbs();
        GraphProcessor::track_function_start(function_name, input_parameters);
        open.push_back(n_starts++);
        if (open.size() == 1 && function_name.starts_with("ggml::")) {
            node = node_l1{.start = open.back(), .peak_bytes = l1_bytes};
        }
    }

    void track_function_end() override {
        replay_cbs();
        GraphProcessor::track_function_end();
        ended();
    }

    void track_function_end(const std::any & output) override {
        replay_cbs();
        GraphProcessor::track_function_end(output);
        ended();
    }

    std::size_t depth() const { return open.size(); }

    // ttnn's launch() opens a function before anything that can throw and closes it only on success, so an exception
    // caught mid-operation (ttprm_bridge falling back to plain ttnn) leaves it open. Its device operation never ran,
    // so the device profiler has no row for it either.
    void close_unfinished(std::size_t depth) {
        while (open.size() > depth) {
            unfinished_starts.insert(open.back());
            unfinished_ends.insert(n_ends);
            track_function_end();
        }
    }

    nlohmann::json report() const {
        nlohmann::json report = get_report();
        std::unordered_map<std::size_t, std::size_t> peak_starts;
        std::unordered_map<std::size_t, std::size_t> peak_ends;
        for (std::size_t i = 0; i < l1_peaks.size(); ++i) {
            peak_starts[l1_peaks[i].start] = i;
            peak_ends[l1_peaks[i].end] = i;
        }
        std::vector<std::pair<uint64_t, uint64_t>> peak_counters(l1_peaks.size());
        std::size_t starts = 0;
        std::size_t ends = 0;
        for (auto & node : report["graph"]) {
            const auto type = node.value("node_type", "");
            if (type == "function_start") {
                const std::size_t ordinal = starts++;
                // ttnn-visualizer links a memory report to a performance report only if the device operations
                // recorded here match the profiler's rows one to one. It skips names containing "::".
                if (unfinished_starts.contains(ordinal)) {
                    rename_unfinished(node);
                }
                if (auto it = peak_starts.find(ordinal); it != peak_starts.end()) {
                    peak_counters[it->second].first = node.value("counter", 0);
                }
            } else if (type == "function_end") {
                const std::size_t ordinal = ends++;
                if (unfinished_ends.contains(ordinal)) {
                    rename_unfinished(node);
                }
                if (auto it = peak_ends.find(ordinal); it != peak_ends.end()) {
                    peak_counters[it->second].second = node.value("counter", 0);
                }
            }
        }

        // graph_report.py gives each listed address the latest page snapshot at or before the operation's end, which
        // for a peak buffer freed inside the node would be whatever took its address later. Pin the peak buffers'
        // pages at the node's end, and put back what is live there right after.
        auto & pages = report["buffer_pages_by_address"];
        auto add_pages = [&](const buffer_record & buffer, uint64_t counter) {
            if (auto it = peak_chunks.find(buffer.allocation); it != peak_chunks.end()) {
                pages[std::to_string(buffer.address)].push_back(
                    {{"alloc_counter", counter}, {"pages", chunk_pages(it->second)}});
            }
        };
        for (std::size_t i = 0; i < l1_peaks.size(); ++i) {
            const auto & peak = l1_peaks[i];
            const auto [start, end] = peak_counters[i];
            auto & buffers = report["per_operation_buffers"][std::to_string(start)];
            buffers = nlohmann::json::array();
            for (const auto & buffer : peak.dram) {
                buffers.push_back(buffer_json(buffer));
            }
            for (const auto & buffer : peak.l1) {
                buffers.push_back(buffer_json(buffer));
                add_pages(buffer, end);
            }
            for (const auto & buffer : peak.restore) {
                add_pages(buffer, end + 1);
            }
        }
        if (pages.is_null()) {
            report.erase("buffer_pages_by_address");
        }
        return report;
    }

private:
    // graph_report.py lists the buffers live when each top-level operation ends. Everything a node puts in L1 is
    // scratch for the ttnn operations inside it and gone by then, so a node that used L1 is listed at its L1 peak.
    struct node_l1 {
        std::size_t start = 0;
        uint64_t peak_bytes = 0;
        std::vector<buffer_record> peak;
    };

    struct l1_peak {
        std::size_t start;  // function_start ordinal
        std::size_t end;    // function_end ordinal
        std::vector<buffer_record> dram;
        std::vector<buffer_record> l1;
        // Buffers at a peak buffer's address that are live when the node ends
        std::vector<buffer_record> restore;
    };

    const tt::tt_metal::IDevice * on_mesh(const tt::tt_metal::IDevice * device) const {
        return device == chip ? &mesh : device;
    }

    void replay_cbs() {
        if (!pending_replay) {
            return;
        }
        pending_replay = false;
        for (const auto & cb : *recording) {
            GraphProcessor::track_allocate_cb(cb.core_ranges, cb.address, cb.size, false, &mesh);
        }
    }

    // The first and last page of each bank: all graph_report.py keeps of them is their extent and the largest page.
    // num_pages, which ttnn-visualizer never reads, comes out as the number of pages written.
    static nlohmann::json chunk_pages(const l1_allocation & allocation) {
        nlohmann::json pages = nlohmann::json::array();
        auto page = [&](const bank_chunk & chunk, uint64_t page_address) {
            pages.push_back({{"device_id", allocation.device_id},
                             {"address", allocation.address},
                             {"core_y", chunk.core.y},
                             {"core_x", chunk.core.x},
                             {"bank_id", chunk.bank_id},
                             {"page_index", 0},
                             {"page_address", page_address},
                             {"page_size", allocation.page_size},
                             {"buffer_type", allocation.buffer_type}});
        };
        for (const auto & chunk : allocation.chunks) {
            page(chunk, chunk.start);
            if (chunk.pages > 1) {
                page(chunk, chunk.end - allocation.page_size);
            }
        }
        return pages;
    }

    static void rename_unfinished(nlohmann::json & node) {
        auto & name = node["params"]["name"];
        name = name.get<std::string>() + "::unfinished";
    }

    void ended() {
        if (!open.empty()) {
            open.pop_back();
        }
        ++n_ends;
        if (!open.empty() || !node) {
            return;
        }
        // A node that only freed L1 it did not allocate has no peak above its start; its end state says it all
        if (node->peak_bytes > l1_bytes && !node->peak.empty()) {
            l1_peak peak{.start = node->start, .end = n_ends - 1, .l1 = std::move(node->peak)};
            peak.dram.reserve(live_dram.size());
            for (const auto & [address, live] : live_dram) {
                peak.dram.push_back(live);
            }
            for (const auto & buffer : peak.l1) {
                keep_chunks(buffer.allocation);
                if (auto it = live_l1.find(buffer.address); it != live_l1.end() && it->second.allocation != buffer.allocation) {
                    peak.restore.push_back(it->second);
                    keep_chunks(it->second.allocation);
                }
            }
            l1_peaks.push_back(std::move(peak));
        }
        node.reset();
        // A freed buffer can still be in the peak of the node running when it was freed, and no later one
        std::erase_if(undecided_chunks, [&](const auto & entry) {
            const auto it = live_l1.find(entry.second.address);
            return it == live_l1.end() || it->second.allocation != entry.first;
        });
    }

    void keep_chunks(std::size_t allocation) {
        if (auto it = undecided_chunks.find(allocation); it != undecided_chunks.end()) {
            peak_chunks.insert(undecided_chunks.extract(it));
        }
    }

    tt::tt_metal::distributed::MeshDevice & mesh;
    const tt::tt_metal::IDevice * chip;
    std::vector<std::size_t> open;
    std::size_t n_starts = 0;
    std::size_t n_ends = 0;
    std::size_t n_allocations = 0;
    std::unordered_set<std::size_t> unfinished_starts;
    std::unordered_set<std::size_t> unfinished_ends;
    std::vector<cb_allocation> * recording = nullptr;
    bool pending_replay = false;
    std::unordered_map<uint64_t, buffer_record> live_dram;
    std::unordered_map<uint64_t, buffer_record> live_l1;
    uint64_t l1_bytes = 0;
    std::optional<node_l1> node;
    std::vector<l1_peak> l1_peaks;
    // Per-bank extent of L1 allocations, by allocation ordinal: those still able to become part of a peak, and those
    // a peak refers to
    std::unordered_map<std::size_t, l1_allocation> undecided_chunks;
    std::unordered_map<std::size_t, l1_allocation> peak_chunks;
};

using config_attributes = ttnn::Config::attributes_t;

// graph_report.py shifts the operation ids of every further capture file in a directory by 10000, and a file holding
// more top-level operations than that silently overwrites the next file's rows. Frees outside any operation can still
// add synthetic ttnn::deallocate operations, so rotate the capture well before that.
constexpr int k_max_ops_per_capture = 8000;

bool logging_on = false;
bool graphs_on = false;
bool detailed_buffers_on = false;

struct recorder {
    std::recursive_mutex mutex;
    std::shared_ptr<ggml_metalium_graph_processor> capture;
    int n_ops = 0;
    // A result aliasing a source is reported under a fresh id, which its later readers must report too, in whichever
    // capture file: graph_report.py merges them. Keyed by the stored tensor object as well, since a pass-through copy
    // of a tensor carries the same id.
    std::map<std::pair<const ttnn::Tensor *, std::uint64_t>, std::uint64_t> renamed;
};

// Leaked: backend calls can still arrive while static objects are destroyed at exit
recorder & the_recorder() {
    static auto * r = new recorder;
    return *r;
}

// This thread's nested calls into the backend while recording
thread_local int entries = 0;

std::once_flag report_dir_once;
std::optional<std::filesystem::path> report_dir;
std::atomic<unsigned> next_capture_index{0};

// Mirrors ttnn.load_config_from_dictionary for one entry.
bool set_config_entry(const std::string & key, const nlohmann::ordered_json & value) {
    bool found = false;
    reflect::for_each<config_attributes>([&](auto I) {
        constexpr std::size_t index = decltype(I)::value;
        if (found || reflect::member_name<index, config_attributes>() != key) {
            return;
        }
        found = true;
        using T = std::decay_t<decltype(ttnn::CONFIG.get<index>())>;
        if constexpr (std::is_same_v<T, std::optional<std::filesystem::path>>) {
            ttnn::CONFIG.set<index>(value.is_null() ? T{} : T{value.get<std::string>()});
        } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
            ttnn::CONFIG.set<index>(T{value.get<std::string>()});
        } else {
            ttnn::CONFIG.set<index>(value.get<T>());
        }
    });
    return found;
}

// The same file ttnn.save_config_to_json_file writes next to a report.
nlohmann::ordered_json config_to_json() {
    nlohmann::ordered_json config = nlohmann::ordered_json::object();
    reflect::for_each<config_attributes>([&](auto I) {
        constexpr std::size_t index = decltype(I)::value;
        const std::string name(reflect::member_name<index, config_attributes>());
        const auto value = ttnn::CONFIG.get<index>();
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::optional<std::filesystem::path>>) {
            config[name] = value ? nlohmann::ordered_json(value->string()) : nlohmann::ordered_json(nullptr);
        } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
            config[name] = value.string();
        } else {
            config[name] = value;
        }
    });
    const auto report_path = ttnn::CONFIG.get<"report_path">();
    config["report_path"] = report_path ? nlohmann::ordered_json(report_path->string()) : nlohmann::ordered_json(nullptr);
    return config;
}

bool prepare_report_dir() {
    std::call_once(report_dir_once, [] {
        const std::filesystem::path path = *ttnn::CONFIG.get<"report_path">();
        std::error_code ec;
        std::filesystem::create_directories(path, ec);
        if (ec) {
            log_warning(tt::LogOp, "ggml-metalium: cannot create graph report directory {}: {}", path.string(), ec.message());
            return;
        }

        // Runs started within the same minute share a directory, and graph_report.py merges every capture in it
        std::vector<std::filesystem::path> stale;
        for (auto it = std::filesystem::directory_iterator(path, ec); !ec && it != std::filesystem::directory_iterator();
             it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.starts_with("graph_capture_") || name == "db.sqlite") {
                stale.push_back(it->path());
            }
        }
        for (const auto & file : stale) {
            std::filesystem::remove(file, ec);
        }

        std::ofstream config_file(path / "config.json");
        config_file << config_to_json().dump(4);
        if (!config_file) {
            log_warning(tt::LogOp, "ggml-metalium: cannot write {}", (path / "config.json").string());
            return;
        }

        report_dir = path;
        log_info(tt::LogOp,
                 "ggml-metalium: writing graph captures to {}. Convert them for ttnn-visualizer with tt-metal's Python "
                 "environment: python $TT_METAL_HOME/ttnn/ttnn/graph_report.py {} {}",
                 path.string(), path.string(), path.string());
    });
    return report_dir.has_value();
}

void write_capture(const ggml_metalium_graph_processor & capture) {
    const auto path = *report_dir / fmt::format("graph_capture_g{:06}.json", next_capture_index++);
    // graph_report.py imports every *.json in the directory and gives up on the first one it cannot parse, so only a
    // complete capture may take that name
    auto partial = path;
    partial += ".partial";
    try {
        const std::string text = capture.report().dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        std::ofstream file(partial, std::ios::binary | std::ios::trunc);
        file << text;
        file.close();
        if (!file) {
            throw std::runtime_error("write failed");
        }
        std::filesystem::rename(partial, path);
    } catch (const std::exception & e) {
        std::error_code ec;
        std::filesystem::remove(partial, ec);
        log_warning(tt::LogOp, "ggml-metalium: failed to write graph capture {}: {}", path.string(), e.what());
    }
}

void begin_capture(recorder & r, tt::tt_metal::distributed::MeshDevice & device) {
    auto capture = std::make_shared<ggml_metalium_graph_processor>(device);

    // Buffers allocated before the capture never reach it. Record them as allocated when the capture starts, before
    // detailed tracing would walk every page of every buffer for each of them.
    std::vector<const tt::tt_metal::Buffer *> live;
    for (const tt::tt_metal::Buffer * buffer : device.allocator()->get_allocated_buffers()) {
        live.push_back(buffer);
    }
    std::ranges::sort(live, {}, [](const tt::tt_metal::Buffer * buffer) {
        return std::pair(buffer->is_dram(), buffer->address());
    });
    for (const tt::tt_metal::Buffer * buffer : live) {
        capture->track_allocate(buffer);
    }

    if (detailed_buffers_on) {
        ttnn::graph::GraphProcessor::enable_detailed_buffer_tracing();
    }
    r.capture = std::move(capture);
    r.n_ops = 0;
}

void end_capture(recorder & r) {
    auto capture = std::exchange(r.capture, nullptr);
    if (!capture) {
        return;
    }
    try {
        capture->end_capture();
    } catch (const std::exception & e) {
        log_warning(tt::LogOp, "ggml-metalium: failed to finish graph capture: {}", e.what());
    }
    if (r.n_ops > 0) {
        write_capture(*capture);
    }
    // Like ttnn's report fixture, only after writing: get_report() leaves out the buffer_pages snapshot once it is off
    if (detailed_buffers_on) {
        ttnn::graph::GraphProcessor::disable_detailed_buffer_tracing();
    }
}

void pop_capture(recorder & r) {
    auto & tracker = tt::tt_metal::GraphTracker::instance();
    if (!tracker.get_processors().empty() && tracker.get_processors().back() == r.capture) {
        tracker.pop_processor();
    } else {
        log_warning(tt::LogOp, "ggml-metalium: another graph processor was pushed over the capture; it stays active");
    }
}

}  // namespace

void ggml_metalium_report_init(bool memory_profile) {
    if (memory_profile) {
        // In this order Config::validate never sees logging enabled in fast runtime mode
        ttnn::CONFIG.set<"enable_fast_runtime_mode">(false);
        ttnn::CONFIG.set<"enable_logging">(true);
        ttnn::CONFIG.set<"report_name">(std::optional<std::filesystem::path>("ggml-metalium"));
        ttnn::CONFIG.set<"enable_graph_report">(true);
    }

    // ttnn's Python package is the only reader of this variable; C++ hosts otherwise keep the defaults
    if (const char * overrides = std::getenv("TTNN_CONFIG_OVERRIDES")) {
        const auto parsed = nlohmann::ordered_json::parse(overrides, nullptr, false);
        if (!parsed.is_object()) {
            GGML_ABORT("TTNN_CONFIG_OVERRIDES is not a JSON object");
        }
        for (const auto & [key, value] : parsed.items()) {
            bool found = false;
            try {
                found = set_config_entry(key, value);
            } catch (const nlohmann::json::exception & e) {
                GGML_ABORT("Invalid value for %s in TTNN_CONFIG_OVERRIDES: %s", key.c_str(), e.what());
            }
            if (!found) {
                GGML_ABORT("Unknown configuration key in TTNN_CONFIG_OVERRIDES: %s", key.c_str());
            }
        }
    }

    // ttnn stamps the report directory name with the time report_path is first read
    (void)ttnn::CONFIG.get<"report_path">();

    const auto report_name = ttnn::CONFIG.get<"report_name">();
    logging_on = ttnn::CONFIG.get<"enable_logging">();
    graphs_on = logging_on && ttnn::CONFIG.get<"enable_graph_report">() && report_name && !report_name->empty();
    detailed_buffers_on = ttnn::CONFIG.get<"enable_detailed_buffer_report">();
}

void ggml_metalium_report_flush() {
    if (!graphs_on) {
        return;
    }
    auto & r = the_recorder();
    std::lock_guard<std::recursive_mutex> guard(r.mutex);
    if (entries == 0) {
        end_capture(r);
    }
}

ggml_metalium_report::ggml_metalium_report(tt::tt_metal::distributed::MeshDevice & device) :
    device(device),
    logging(logging_on) {
    if (!graphs_on || !prepare_report_dir()) {
        return;
    }
    auto & r = the_recorder();
    lock = std::unique_lock<std::recursive_mutex>(r.mutex);
    recording = true;
    if (entries++ == 0) {
        if (!r.capture) {
            begin_capture(r, device);
        }
        tt::tt_metal::GraphTracker::instance().push_processor(r.capture);
    }
}

ggml_metalium_report::~ggml_metalium_report() {
    if (!recording || --entries > 0) {
        return;
    }
    auto & r = the_recorder();
    // An exception left the call mid-operation; later calls must not nest inside it
    if (r.capture->depth() > 0) {
        log_warning(tt::LogOp, "ggml-metalium: {} ended with {} unfinished operation(s) in the graph capture", op_name,
                    r.capture->depth());
        r.capture->close_unfinished(0);
    }
    pop_capture(r);
    if (flush) {
        end_capture(r);
    }
}

void ggml_metalium_report::op_begin(const std::string & name, const char * tensor_name,
                                    std::vector<std::reference_wrapper<const ttnn::Tensor>> & inputs) {
    op_name = name;
    if (recording) {
        auto & r = the_recorder();
        if (r.capture->depth() == 0) {
            // Counted on start so that a capture whose first operation throws is still written, showing it incomplete
            ++r.n_ops;
        }
        input_ids.clear();
        std::vector<ttnn::Tensor> renamed_inputs;
        renamed_inputs.reserve(inputs.size());
        std::vector<std::reference_wrapper<const ttnn::Tensor>> reported;
        for (const ttnn::Tensor & t : inputs) {
            input_ids.push_back(t.tensor_id);
            if (auto it = r.renamed.find({&t, t.tensor_id}); it != r.renamed.end()) {
                renamed_inputs.push_back(t);
                renamed_inputs.back().tensor_id = it->second;
                reported.emplace_back(renamed_inputs.back());
            } else {
                reported.emplace_back(t);
            }
        }
        std::string tensor(tensor_name);
        tt::tt_metal::GraphTracker::instance().track_function_start(op_name, tensor, reported);
        op_depth = r.capture->depth();
    }
    if (logging) {
        synchronize();
        tt::LoggerRegistry::instance().get(tt::LogOp)->debug("Started {:50}", op_name);
    }
}

void ggml_metalium_report::op_end(const ttnn::Tensor * output) {
    if (logging) {
        synchronize();
        tt::LoggerRegistry::instance().get(tt::LogOp)->debug("Finished {:50}", op_name);
    }
    if (!recording) {
        return;
    }
    auto & r = the_recorder();
    if (r.capture->depth() > op_depth) {
        log_warning(tt::LogOp, "ggml-metalium: {} left {} ttnn operation(s) unfinished in the graph capture", op_name,
                    r.capture->depth() - op_depth);
        r.capture->close_unfinished(op_depth);
    }
    auto & tracker = tt::tt_metal::GraphTracker::instance();
    if (output != nullptr) {
        // ttnn's Python frontend gives every result a fresh id. A result aliasing a source (in-place SET,
        // pass-through nodes) gets one too, on a copy: the tensor object itself may be the source's, even a weight's.
        ttnn::Tensor result = *output;
        if (std::ranges::find(input_ids, result.tensor_id) != input_ids.end()) {
            result.tensor_id = ttnn::Tensor::next_tensor_id();
            r.renamed[{output, output->tensor_id}] = result.tensor_id;
        }
        tracker.track_function_end(result);
    } else {
        tracker.track_function_end();
    }
    if (r.capture->depth() == 0 && r.n_ops >= k_max_ops_per_capture) {
        pop_capture(r);
        end_capture(r);
        begin_capture(r, device);
        tracker.push_processor(r.capture);
    }
}

void ggml_metalium_report::synchronize() {
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
}

 // This file is like test-backend-ops.cpp but we expect _everything_ to be supported by the Metalium backend.
// Also tests for edge cases in Metalium. (ex: Metalium/TTNN nativly uses 32x32 matrices as it's smallest unit)

// some code stolen from test-backend-ops.cpp
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <functional>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

#if !defined (GGML_USE_METALIUM)
    #error "This file should only be compiled with Metalium backend enabled"
#endif

#include <ggml-metalium.h>

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    const ggml_type_traits* tt = ggml_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt->to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ASSERT(false);
                    }
                }
            }
        }
    }

    return tv;
}

static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static double pcc(const float * a, const float * b, size_t n) {
    // Calculate the mean of x and y values
    double a_mean = 0.0;
    double b_mean = 0.0;

    for (size_t i = 0; i < n; i++) {
        a_mean += a[i];
        b_mean += b[i];
    }

    a_mean /= n;
    b_mean /= n;

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float x_diff = a[i] - a_mean;
        float y_diff = b[i] - b_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= n;
    x_stddev /= n;
    y_stddev /= n;

    // Calculate the correlation coefficient
    double correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

static bool isinf_or_max(float f) {
    return std::isinf(f) || f == std::numeric_limits<float>::max() || f == -std::numeric_limits<float>::max();
}

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    static std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(min, max);
    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);

    for (size_t i = 0; i < size; i++) {
        data[i] = distribution(generator);
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(size % ggml_blck_size(tensor->type) == 0);
        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, size));
        std::vector<float> imatrix(tensor->ne[0], 1.0f); // dummy importance matrix
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }
        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, size/tensor->ne[0], tensor->ne[0], im);
        GGML_ASSERT(ggml_validate_row_data(tensor->type, dataq.data(), dataq.size()));
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> datai32(size);
        std::uniform_int_distribution<int32_t> distribution_int32(0, 2048);
        for (size_t i = 0; i < size; i++) {
            datai32[i] = distribution_int32(generator);
        }
        ggml_backend_tensor_set(tensor, datai32.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ASSERT(false);
    }
}

static void initialize_tensors(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        init_tensor_uniform(t);
    }
}

enum class TestResult {
    OK,
    FAIL,
    NOT_SUPPORTED
};

struct test_case
{
    test_case(std::string name, std::function<ggml_tensor* (ggml_context*)> build_graph, const std::function<double(const float*, const float*, size_t n)>& loss = nmse)
        : name(std::move(name)), loss(loss), build_graph(std::move(build_graph)) {}
    std::string name;
    float max_err = 1e-4;
    std::function<double(const float*, const float*, size_t n)> loss;
    std::function<ggml_tensor* (ggml_context*)> build_graph;

    static const int sentinel_size = 1024;
    std::vector<ggml_tensor *> sentinels;
    ggml_cgraph * gf = nullptr;

    void add_sentinel(ggml_context * ctx) {
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    TestResult eval(ggml_backend_t backend1, ggml_backend_t backend2) {
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        gf = ggml_new_graph(ctx);

        // pre-graph sentinel
        add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);

        printf("  %s (%s): ", name.c_str(), ggml_op_desc(out));
        if(out->op == GGML_OP_NONE) {
            printf("\033[1;31mTEST_ERROR\033[0m operator should not be NONE. Test is buggy\n");
            return TestResult::FAIL;
        }
        fflush(stdout);

        // check if the backends support the ops
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    printf("\033[1;33mNOT_SUPPORTED\033[0m by [%s]. Rejected OP: %s\n", ggml_backend_name(backend), ggml_op_desc(t));
                    ggml_free(ctx);
                    return TestResult::NOT_SUPPORTED;
                }
            }
        }
        // post-graph sentinel
        add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);
        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
            ggml_free(ctx);
            return TestResult::FAIL;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            ggml_graph_add_node(gf, sentinel);
        }

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            bool   ok;
            double max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
            std::function<double(const float*, const float*, size_t n)> loss;
        };

        callback_userdata ud {
            true,
            max_err,
            backend1,
            backend2,
            loss
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = ud->loss(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                printf("[%s / %s] loss = %.9f > %.9f ", ggml_op_desc(t1), t1->name, err, ud->max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud, &out, 1);

        if (!cmp_ok) {
            printf("compare failed ");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        if (ud.ok && cmp_ok) {
            printf("\033[1;32mOK\033[0m\n");
            return TestResult::OK;
        }

        printf("\033[1;31mFAIL\033[0m\n");
        return TestResult::FAIL;
    }
};

static std::unique_ptr<test_case> make_test(const std::function<ggml_tensor* (ggml_context*)> & build_graph, std::string name, float max_err = 1e-4) {
    std::unique_ptr<test_case> tc = std::make_unique<test_case>(std::move(name), build_graph);
    tc->max_err = max_err;
    return tc;
}

static std::string type_name(ggml_type type)
{
    return ggml_get_type_traits(type)->type_name;
}

// CPU reference implementation of Flux-RoPE (interleaved)
// PE format: [D, L] in GGML ne, with pe[l, 2*p] = cos, pe[l, 2*p+1] = -sin
// x format:  [D, L, B] in GGML ne
// Rotation (standard +theta): x' = x*cos - y*sin, y' = x*sin + y*cos
//   with c=cos, ns=-sin:      x' = x*c + y*ns,    y' = y*c - x*ns
static void flux_rope_cpu_impl(ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)ith; (void)nth; (void)userdata;

    const ggml_tensor* x  = dst->src[0];
    const ggml_tensor* pe = dst->src[1];

    const int64_t D = x->ne[0];
    const int64_t L = x->ne[1];
    const int64_t B = x->ne[2];

    const float* x_data  = (const float*)x->data;
    const float* pe_data = (const float*)pe->data;
    float* dst_data      = (float*)dst->data;

    for (int64_t b = 0; b < B; b++) {
        for (int64_t l = 0; l < L; l++) {
            for (int64_t p = 0; p < D / 2; p++) {
                const float xe = x_data[b*L*D + l*D + 2*p];
                const float xo = x_data[b*L*D + l*D + 2*p + 1];
                const float c  = pe_data[l*D + 2*p];       // cos(theta)
                const float ns = pe_data[l*D + 2*p + 1];   // -sin(theta)

                dst_data[b*L*D + l*D + 2*p]     = xe * c + xo * ns;  // xe*cos - xo*sin
                dst_data[b*L*D + l*D + 2*p + 1] = xo * c - xe * ns;  // xe*sin + xo*cos
            }
        }
    }
}

// Magic tag for identifying Flux-RoPE custom ops in the metalium backend
// 0x464C5530 = interleaved
static constexpr uintptr_t FLUX_ROPE_INTERLEAVED_TAG = 0x464C5530;

static void add_flux_rope_tests(std::vector<std::unique_ptr<test_case>>& tests)
{
    // Flux-RoPE interleaved: tile-aligned dimensions
    tests.push_back(make_test([](ggml_context* ctx) {
        const int64_t D = 128, L = 64, B = 8;
        ggml_tensor* x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, B);
        ggml_tensor* pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L);
        ggml_tensor* args[2] = {x, pe};
        return ggml_custom_4d(ctx, GGML_TYPE_F32, D, L, B, 1,
            args, 2, flux_rope_cpu_impl, 1, (void*)FLUX_ROPE_INTERLEAVED_TAG);
    }, "Flux-RoPE interleaved 128x64x8", 1e-3));

    // Flux-RoPE: small tile-aligned
    tests.push_back(make_test([](ggml_context* ctx) {
        const int64_t D = 32, L = 32, B = 1;
        ggml_tensor* x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, B);
        ggml_tensor* pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L);
        ggml_tensor* args[2] = {x, pe};
        return ggml_custom_4d(ctx, GGML_TYPE_F32, D, L, B, 1,
            args, 2, flux_rope_cpu_impl, 1, (void*)FLUX_ROPE_INTERLEAVED_TAG);
    }, "Flux-RoPE interleaved 32x32x1", 1e-3));

    // Flux-RoPE: non-tile-aligned D and L (Flux-style d_head=60)
    tests.push_back(make_test([](ggml_context* ctx) {
        const int64_t D = 60, L = 50, B = 4;
        ggml_tensor* x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, B);
        ggml_tensor* pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L);
        ggml_tensor* args[2] = {x, pe};
        return ggml_custom_4d(ctx, GGML_TYPE_F32, D, L, B, 1,
            args, 2, flux_rope_cpu_impl, 1, (void*)FLUX_ROPE_INTERLEAVED_TAG);
    }, "Flux-RoPE interleaved 60x50x4 (non tile aligned)", 1e-3));

    // Flux-RoPE: large batch (typical Flux: B = N*n_head = 2*24 = 48)
    tests.push_back(make_test([](ggml_context* ctx) {
        const int64_t D = 128, L = 256, B = 48;
        ggml_tensor* x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, B);
        ggml_tensor* pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L);
        ggml_tensor* args[2] = {x, pe};
        return ggml_custom_4d(ctx, GGML_TYPE_F32, D, L, B, 1,
            args, 2, flux_rope_cpu_impl, 1, (void*)FLUX_ROPE_INTERLEAVED_TAG);
    }, "Flux-RoPE interleaved 128x256x48 (Flux-scale)", 1e-3));

    // Flux-RoPE: single pair (D=2), edge case
    tests.push_back(make_test([](ggml_context* ctx) {
        const int64_t D = 2, L = 64, B = 1;
        ggml_tensor* x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, B);
        ggml_tensor* pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L);
        ggml_tensor* args[2] = {x, pe};
        return ggml_custom_4d(ctx, GGML_TYPE_F32, D, L, B, 1,
            args, 2, flux_rope_cpu_impl, 1, (void*)FLUX_ROPE_INTERLEAVED_TAG);
    }, "Flux-RoPE interleaved 2x64x1 (minimal D)", 1e-3));

    // --- Non-interleaved tests ---
    // These mirror apply_rope's non-interleaved path: interleave x, run fused
    // kernel, un-interleave output.  The graph is executed on both CPU (via the
    // function-pointer fallback) and metalium (via the tag dispatch), so
    // comparing the two validates the full pipeline.

    auto make_non_interleaved_test = [&](int64_t D, int64_t L, int64_t B, const char* name) {
        tests.push_back(make_test([=](ggml_context* ctx) {
            // x in non-interleaved layout: [D, L, B]
            // Data order per row: [x0, x1, ..., x_{D/2-1}, y0, y1, ..., y_{D/2-1}]
            ggml_tensor* x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, B);
            ggml_tensor* pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, L);

            // Interleave x: [D, L, B] -> [D/2, 2, L, B] -> permute(1,0,2,3) -> [2, D/2, L, B] -> cont -> [D, L, B]
            auto x_prep = ggml_reshape_4d(ctx, x, D / 2, 2, L, B);
            x_prep = ggml_cont(ctx, ggml_permute(ctx, x_prep, 1, 0, 2, 3));
            x_prep = ggml_reshape_3d(ctx, x_prep, D, L, B);

            ggml_tensor* args[2] = {x_prep, pe};
            auto out = ggml_custom_4d(ctx, GGML_TYPE_F32, D, L, B, 1,
                args, 2, flux_rope_cpu_impl, 1, (void*)FLUX_ROPE_INTERLEAVED_TAG);

            // Un-interleave output: [D, L, B] -> [2, D/2, L, B] -> permute(1,0,2,3) -> [D/2, 2, L, B] -> cont -> [D, L, B]
            out = ggml_reshape_4d(ctx, out, 2, D / 2, L, B);
            out = ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));
            out = ggml_reshape_3d(ctx, out, D, L, B);
            return out;
        }, name, 1e-3));
    };

    make_non_interleaved_test(128, 64, 8,  "Flux-RoPE non-interleaved 128x64x8");
    make_non_interleaved_test(32,  32, 1,  "Flux-RoPE non-interleaved 32x32x1");
    make_non_interleaved_test(60,  50, 4,  "Flux-RoPE non-interleaved 60x50x4 (non tile aligned)");
    make_non_interleaved_test(128, 256, 48, "Flux-RoPE non-interleaved 128x256x48 (Flux-scale)");
    make_non_interleaved_test(2,   64, 1,  "Flux-RoPE non-interleaved 2x64x1 (minimal D)");
}

// Mirror the FULL Rope::apply_rope (interleaved) graph, INCLUDING the on-device
// pe-extraction (strided ggml_view_3d row-0 of a real [2,2,d/2,L] pe tensor + cont)
// and the x-prep permute/cont that the bare flux_rope tests above skip.  The harness
// compares every node CPU-vs-Metalium, so a divergence localizes to the exact prep op
// or the kernel.  Shapes match z-image (d_head=128, num_heads=30).
static void add_apply_rope_full_path_tests(std::vector<std::unique_ptr<test_case>>& tests)
{
    // Isolated pe-extraction: [2,2,d/2,L] -> row-0 strided view -> cont -> [D,L].
    // This is the part the existing flux_rope tests never exercise on device.
    auto make_pe_extract = [&](int64_t d_head, int64_t L, const char* name) {
        tests.push_back(make_test([=](ggml_context* ctx) {
            ggml_tensor* pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, d_head / 2, L);
            auto pe_prep = ggml_view_3d(ctx, pe, 2, d_head / 2, L, pe->nb[2], pe->nb[3], 0);
            pe_prep = ggml_cont(ctx, pe_prep);
            return ggml_reshape_2d(ctx, pe_prep, d_head, L);
        }, name, 1e-3));
    };
    make_pe_extract(128, 64,   "apply_rope pe-extract d128 L64");
    make_pe_extract(128, 256,  "apply_rope pe-extract d128 L256");
    make_pe_extract(128, 1024, "apply_rope pe-extract d128 L1024 (z-image scale)");

    // Full path: x [d_head, n_head, L, N] + pe [2,2,d/2,L] -> apply_rope (interleaved).
    auto make_full = [&](int64_t d_head, int64_t n_head, int64_t L, int64_t N, const char* name) {
        tests.push_back(make_test([=](ggml_context* ctx) {
            ggml_tensor* x  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_head, n_head, L, N);
            ggml_tensor* pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, d_head / 2, L);

            auto x_prep = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [d_head, L, n_head, N]
            x_prep = ggml_reshape_3d(ctx, x_prep, d_head, L, n_head * N);    // [D, L, B]

            auto pe_prep = ggml_view_3d(ctx, pe, 2, d_head / 2, L, pe->nb[2], pe->nb[3], 0);
            pe_prep = ggml_cont(ctx, pe_prep);
            pe_prep = ggml_reshape_2d(ctx, pe_prep, d_head, L);

            ggml_tensor* args[2] = {x_prep, pe_prep};
            auto x_out = ggml_custom_4d(ctx, GGML_TYPE_F32, d_head, L, n_head * N, 1,
                args, 2, flux_rope_cpu_impl, 1, (void*)FLUX_ROPE_INTERLEAVED_TAG);
            return ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);
        }, name, 1e-3));
    };
    make_full(128, 4,  64,   1, "apply_rope full path d128 h4 L64");
    make_full(128, 30, 256,  1, "apply_rope full path d128 h30 L256 (z-image heads)");
    make_full(128, 30, 1024, 1, "apply_rope full path d128 h30 L1024 (z-image scale)");
}

// The real diffusion weights are BF16 (z-image-turbo-BF16.gguf); every Linear is
// mul_mat(bf16_weight, f32_activation).  The existing mul_mat tests are all F32, so a
// BF16-weight-specific matmul bug would pass every F32 unit test yet corrupt the whole
// model.  Compare BF16/F16-weight matmul (incl. large activations) against the CPU ref.
static void add_lowprec_matmul_tests(std::vector<std::unique_ptr<test_case>>& tests)
{
    auto make = [&](int64_t K, int64_t M, int64_t N, ggml_type wtype, float scale, std::string name) {
        tests.push_back(make_test([=](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_2d(ctx, wtype, K, M);          // weight [K, M]
            ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);  // activation [K, N]
            if (scale != 1.0f) {
                b = ggml_scale(ctx, b, scale);
            }
            return ggml_mul_mat(ctx, a, b);                                 // [M, N]
        }, name, 5e-2));
    };
    for (ggml_type wt : {GGML_TYPE_BF16, GGML_TYPE_F16}) {
        std::string t = type_name(wt);
        make(128,  128,  64, wt, 1.0f,   "mul_mat " + t + "-weight 128x128 (small)");
        make(64,   3840, 64, wt, 1.0f,   "mul_mat " + t + "-weight x_embedder 64->3840");
        make(2560, 3840, 64, wt, 1.0f,   "mul_mat " + t + "-weight cap_embedder 2560->3840");
        make(3840, 3840, 64, wt, 1.0f,   "mul_mat " + t + "-weight hidden 3840->3840");
        make(3840, 3840, 64, wt, 100.0f, "mul_mat " + t + "-weight 3840 large-act x100");
    }
}

// Hypothesis: realize_ggml_view resolves a VIEW via view_src, but ggml collapses the
// view_src of a view-of-an-inplace-result PAST the inplace node to the stale base.
// So reading a view (or reshape) of x = ggml_*_inplace(a, b) may return a's pre-op data
// instead of the computed result.  Isolated single ops never expose this; a 34-layer
// graph full of inplace residuals/norms would.  cont() forces materialization so the
// node is actually compared (bare view ops are skipped by the compare harness).
static void add_inplace_view_tests(std::vector<std::unique_ptr<test_case>>& tests)
{
    // view (sub-block) of an inplace ADD result
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* c = ggml_add_inplace(ctx, a, b);
        ggml_tensor* v = ggml_view_2d(ctx, c, 32, 64, c->nb[1], 0);
        return ggml_cont(ctx, v);
    }, "view of add_inplace result", 1e-3));

    // full-size view of an inplace ADD result (exercises the view fast-path)
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* c = ggml_add_inplace(ctx, a, b);
        return ggml_cont(ctx, ggml_view_tensor(ctx, c));
    }, "full view of add_inplace result", 1e-3));

    // view of an inplace MUL result
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* c = ggml_mul_inplace(ctx, a, b);
        ggml_tensor* v = ggml_view_2d(ctx, c, 32, 64, c->nb[1], 0);
        return ggml_cont(ctx, v);
    }, "view of mul_inplace result", 1e-3));

    // reshape of an inplace ADD result (control: realize uses src[0], should pass)
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* c = ggml_add_inplace(ctx, a, b);
        return ggml_cont(ctx, ggml_reshape_2d(ctx, c, 32, 128));
    }, "reshape of add_inplace result (control)", 1e-3));

    // view of a NON-inplace ADD result (control: should pass)
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* c = ggml_add(ctx, a, b);
        ggml_tensor* v = ggml_view_2d(ctx, c, 32, 64, c->nb[1], 0);
        return ggml_cont(ctx, v);
    }, "view of (non-inplace) add result (control)", 1e-3));
}

static void add_unittests(std::vector<std::unique_ptr<test_case>>& tests)
{
    const ggml_unary_op supported_unary_ops[] = {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        GGML_UNARY_OP_STEP, // Not supported by Metalium
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_HARDSWISH,
        GGML_UNARY_OP_HARDSIGMOID,
        GGML_UNARY_OP_EXP,
    };

    // TODO: Add more types
    const ggml_type supported_types[] = {
        GGML_TYPE_F32,
        GGML_TYPE_F16,
        GGML_TYPE_BF16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q4_0
    };

    for(auto type : supported_types) {
        for(auto op : supported_unary_ops) {
            tests.push_back(make_test([op](ggml_context* ctx) {
                ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
                return ggml_unary(ctx, a, op);
            }, "Basic activation function for " + type_name(type), 1e-2));
        }
    }

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 96, 96);
        ggml_tensor* v = ggml_view_2d(ctx, a, 64, 64, a->nb[1], 0);
        return ggml_unary(ctx, v, GGML_UNARY_OP_ABS);
    }, "Activation of view"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* b = ggml_cont(ctx, a);
        return b;
    }, "CONT on real tesnor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 64, 64);
        ggml_tensor* b = ggml_cont(ctx, a);
        return b;
    }, "CONT on integer tesnor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 64, 64, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "No-op view"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View into 2D matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32*32);
        ggml_tensor* view = ggml_view_1d(ctx, a, 32*32, 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View flat buffer into 2D matrix"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32*32);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, 32 * sizeof(float), 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View flat buffer into 2D matrix"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32*32);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, 32 * sizeof(float), 0);
        ggml_tensor* transposed = ggml_transpose(ctx, view);
        ggml_tensor* b = ggml_cont(ctx, transposed);
        return b;
    }, "Transposed flat buffer into 2D matrix"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32 * 32 * 64);
        ggml_tensor* view = ggml_view_3d(ctx, a, 32, 32, 32, 32 * sizeof(float), 32 * 32 * sizeof(float), 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View flat buffer into 3D tensor"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32 * 32 * 64);
        ggml_tensor* view = ggml_view_3d(ctx, a, 32, 32, 32, 32 * sizeof(float), 32 * 32 * sizeof(float), 32 * sizeof(float));
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View flat buffer into 3D tensor with offset"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32 * 32 * 64);
        ggml_tensor* view = ggml_view_3d(ctx, a, 32, 32, 32, 32 * sizeof(float), 32 * 32 * sizeof(float), 0);
        ggml_tensor* transposed = ggml_transpose(ctx, view);
        ggml_tensor* b = ggml_cont(ctx, transposed);
        return b;
    }, "View flat buffer into 3D tensor transposed"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 30, 30, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View into 2D matrix, non tile aligned"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, a->nb[1], ggml_type_size(GGML_TYPE_F32));
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View into 2D matrix with offset"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 48, 1, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "1D view into 2D matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 30, 28, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "Rectangular view into 2D matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        return ggml_transpose(ctx, a);
    }, "transpose 2D square matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 28);
        return ggml_transpose(ctx, a);
    }, "transpose 2D rectangular matrix"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        return ggml_transpose(ctx, a);
    }, "transpose 2D small matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 64, 64);
        return ggml_transpose(ctx, a);
    }, "transpose 3D square matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 64, 64);
        return ggml_cont(ctx, ggml_transpose(ctx, a));
    }, "transpose 3D square matrix"));

    // Failing - need to support tensor copy
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
    //     ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 256, 4, 4, 4);
    //     return ggml_cpy(ctx, a, b);
    // }, "4D tensor copy"));

    // Failing - need to support sum
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 14, 2, 3);
    //     return ggml_sum(ctx, a);
    // }, "sum"));

    // Failing - need to support sum_rows
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 14, 2, 3);
    //     return ggml_sum_rows(ctx, a);
    // }, "sum rows"));

    // Failing
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
    //     ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 16, 1, 4);
    //     return ggml_cpy(ctx, a, b);
    // }, "Copy tensor into tensor of different shape"));

    // FIXME: This sould work but is failing
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
    //     return ggml_view_2d(ctx, ggml_transpose(ctx, a), 4, 12, 4 * 4, 0);
    // }, "View of transposed 4D tensor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 64, 4, 1);
        return ggml_reshape_4d(ctx, a, 32, 128, 4, 1);
    }, "Reshape to tile aligned tensor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 32, 1, 1);
        return ggml_reshape_4d(ctx, a, 16, 32, 2, 1);
    }, "Reshape to non tile aligned tensor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_dup(ctx, a);
    }, "Tensor duplication"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_dup(ctx, ggml_view_tensor(ctx, a));
    }, "Tensor duplication via view"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        ggml_tensor* view = ggml_view_tensor(ctx, a);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_cpy(ctx, view, b);
    }, "Write via view"));
    // Not working yet. Need write support for views
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
    //     ggml_tensor* view = ggml_view_2d(ctx, a, 8, 12, a->nb[1], 1);
    //     ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 12);
    //     return ggml_cpy(ctx, view, b);
    // }, "partial write via view"));
    // TODO: Expend this to attempt all permutations possible

    std::array<int, GGML_MAX_DIMS> permute_order;
    for(int i = 0;i<GGML_MAX_DIMS;i++) {
        permute_order[i] = i;
    }
    do {
        std::string name = "Permutation, order=[";
        for(int i = 0;i<GGML_MAX_DIMS;i++) {
            name += std::to_string(permute_order[i]) + " ";
        }
        name.pop_back();
        name += "]";
        tests.push_back(make_test([permute_order](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 32, 8, 4);
            return ggml_permute(ctx, a, permute_order[0], permute_order[1], permute_order[2], permute_order[3]);
        }, name));
    } while(std::next_permutation(permute_order.begin(), permute_order.end()));

    // (Basics of) what we need to get KV cache working
    // TODO: Map GGML operations into TTNN nlp_kv_cache_load_slice and update_cache_multi_core
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 24);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        return ggml_set_2d(ctx, a, b, b->nb[1], 0);
    }, "Set row of 2D matrix"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 24);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        return ggml_set_2d(ctx, a, b, b->nb[1], a->nb[1]);
    }, "Set row of 2D matrix with offset"));

    // Matrix multiplication
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        return ggml_mul_mat(ctx, a, b);
    }, "2D matrix multiplication"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 128);
        return ggml_mul_mat(ctx, a, b);
    }, "2D matrix multiplication (result non square)"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 38, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 38, 72);
        return ggml_mul_mat(ctx, a, b);
    }, "2D matrix multiplication (result non square, non tile aligned)"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 10);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 10);
        return ggml_mul_mat(ctx, a, b);
    }, "4D matrix multiplication"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 1);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 10);
        return ggml_mul_mat(ctx, a, b);
    }, "4D matrix multiplication with broadcast"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        return ggml_mul_mat(ctx, a, b);
    }, "matrix-vector multiplication"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 18);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24);
        return ggml_mul_mat(ctx, a, b);
    }, "matrix-vector multiplication non tile aligned"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 64);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
        return ggml_add(ctx, a, b);
    }, "Add broadcasted vector to matrix"));

    // TODO: TTNN does not support the style of broadcasting GGML wants
    // Failing
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 20);
    //     ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 10);
    //     return ggml_mul_mat(ctx, a, b);
    // }, "3D matrix multiplication (broadcast)"));

    // Misc
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
        return ggml_clamp(ctx, a, -0.1, 0.25);
    }, "Clamp"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
        return ggml_scale(ctx, a, 2.0);
    }, "Scale"));
    // ???? This should not have worked since I haven't implemented inplace operations
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
        return ggml_scale_inplace(ctx, a, 1.5);
    }, "Scale in place"));
    // RoPE
    for(auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) { // Really a limitation of GGML's CPU implementation - we support more
        tests.push_back(make_test([type](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_3d(ctx, type, 2048, 16, 2);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            return ggml_rope(ctx, a, b, 128, GGML_ROPE_TYPE_NEOX);
        }, "RoPE NEOX " + std::string(ggml_type_name(type))));

        tests.push_back(make_test([type](ggml_context* ctx) {
            float freq_base = 20000.f;
            float freq_scale = 1.4245f;
            float attn_factor = 1.424500f;
            float ext_factor = 0.746500f;
            float beta_fast = 32.f;
            float beta_slow = 1.f;
            ggml_tensor* a = ggml_new_tensor_3d(ctx, type, 2048, 16, 2);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            return ggml_rope_ext(ctx, a, b, NULL, 128, GGML_ROPE_TYPE_NEOX, 512, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        }, "RoPE NEOX " + std::string(ggml_type_name(type)) + " with YaRN"));

        tests.push_back(make_test([type](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_3d(ctx, type, 512, 32, 1);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            return ggml_rope(ctx, a, b, 512, GGML_ROPE_TYPE_NEOX);
        }, "RoPE NEOX in Gemma " + std::string(ggml_type_name(type))));

        tests.push_back(make_test([](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 1);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            return ggml_rope(ctx, a, b, 32, GGML_ROPE_TYPE_NORMAL);
        }, "RoPE Normal " + std::string(ggml_type_name(GGML_TYPE_F32))));
    }

    // TODO: Need a way to inform the RNG
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     int n = 300*256;
    //     int m = 60;
    //     int r = 8;
    //     int be1 = 1;
    //     int be2 = 1;
    //     ggml_tensor * in = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n, m, be1, be2);
    //     ggml_tensor * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, r, be1, be2);
    //     ggml_tensor * out = ggml_get_rows(ctx, in, rows);
    //
    // // DITTO
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor * dst = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 32, 200, 1, 1);

    //     ggml_tensor * src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 1, 1, 1);
    //     ggml_tensor * idx = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1);

    //     ggml_tensor * out = ggml_set_rows(ctx, dst, src, idx);

    //     return out;
    // }, "test MM", 1e-5));

    //     return out;
    // }, "Simple GET_ROWS", 1e-5));
    // more complex tests
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 18);
        ggml_tensor* w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        ggml_tensor* b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_tensor* w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 48);
        ggml_tensor* b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 48);

        ggml_tensor* h1 = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, x), b1));
        ggml_tensor* h2 = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, h1), b2));

        return h2;
    }, "Multi layer perceptron"));

    // Pad operations
    // Basic right-only padding (ggml_pad API pads on the right side of each dim)
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
        return ggml_pad(ctx, a, 32, 32, 0, 0);
    }, "Pad 2D tile aligned"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 30, 28);
        return ggml_pad(ctx, a, 2, 4, 0, 0);
    }, "Pad 2D non tile aligned to tile aligned"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        return ggml_pad(ctx, a, 3, 5, 0, 0);
    }, "Pad 2D non tile aligned result"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        return ggml_pad(ctx, a, 16, 0, 0, 0);
    }, "Pad 1D"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 4);
        return ggml_pad(ctx, a, 0, 0, 2, 0);
    }, "Pad 3D on dim2"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 4, 2);
        return ggml_pad(ctx, a, 8, 8, 2, 1);
    }, "Pad 4D all dims"));

    // ggml_pad_ext: explicit left and right padding per dimension
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
        return ggml_pad_ext(ctx, a, 4, 4, 2, 2, 0, 0, 0, 0);
    }, "Pad ext 2D symmetric"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 18);
        return ggml_pad_ext(ctx, a, 3, 5, 7, 1, 0, 0, 0, 0);
    }, "Pad ext 2D asymmetric non tile aligned"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 3, 2);
        return ggml_pad_ext(ctx, a, 2, 2, 4, 4, 1, 1, 0, 1);
    }, "Pad ext 4D all dims"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
        return ggml_pad_ext(ctx, a, 0, 0, 0, 0, 0, 0, 0, 0);
    }, "Pad ext 2D zero padding (identity)"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
        return ggml_pad_ext(ctx, a, 16, 15, 8, 7, 0, 0, 0, 0);
    }, "Pad ext 2D small tensor large padding"));

    tests.push_back(make_test([](ggml_context* ctx) {
        // A smaller and stripped down version of the MLP Mixer model
        ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
        std::array<ggml_tensor*, 4> h;
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                ggml_tensor* patch = ggml_view_2d(ctx, in, 32, 32, in->nb[1], 32 * y + x * 32);
                h[y * 2 + x] = ggml_relu(ctx, ggml_mul_mat(ctx, w1, patch));
            }
        }
        ggml_tensor* h1 = ggml_concat(ctx, h[0], h[1], 1);
        ggml_tensor* h2 = ggml_concat(ctx, h[2], h[3], 1);
        ggml_tensor* h_all = ggml_concat(ctx, h1, h2, 1);
        return ggml_transpose(ctx, h_all);
    }, "MLP mixer", 1e-3));
}

static void add_im2col_tests(std::vector<std::unique_ptr<test_case>>& tests)
{
    // Helper to create a 2D im2col test
    auto make_im2col_2d_test = [&](int64_t IC, int64_t IH, int64_t IW, int64_t OC,
                                    int64_t KH, int64_t KW, int64_t N,
                                    int s0, int s1, int p0, int p1, int d0, int d1,
                                    const char* name) {
        tests.push_back(make_test([=](ggml_context* ctx) {
            // src0 (kernel): ne [KW, KH, IC, OC]
            ggml_tensor* kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, IC, OC);
            // src1 (image): ne [IW, IH, IC, N]
            ggml_tensor* image  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IW, IH, IC, N);
            return ggml_im2col(ctx, kernel, image, s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F32);
        }, name, 1e-3));
    };

    // Basic 3x3 conv, stride 1, padding 1 (most common in SD VAE)
    make_im2col_2d_test(3, 8, 8, 16, 3, 3, 1,  1, 1, 1, 1, 1, 1,
        "IM2COL 2D 3x3 s1 p1 IC=3 8x8 N=1");

    // 3x3 conv, stride 1, no padding
    make_im2col_2d_test(16, 16, 16, 32, 3, 3, 1,  1, 1, 0, 0, 1, 1,
        "IM2COL 2D 3x3 s1 p0 IC=16 16x16 N=1");

    // 3x3 conv, stride 2, padding 1 (downsampling)
    make_im2col_2d_test(32, 32, 32, 64, 3, 3, 1,  2, 2, 1, 1, 1, 1,
        "IM2COL 2D 3x3 s2 p1 IC=32 32x32 N=1");

    // 1x1 conv (pointwise)
    make_im2col_2d_test(64, 16, 16, 128, 1, 1, 1,  1, 1, 0, 0, 1, 1,
        "IM2COL 2D 1x1 s1 p0 IC=64 16x16 N=1");

    // Batch > 1
    make_im2col_2d_test(3, 8, 8, 16, 3, 3, 4,  1, 1, 1, 1, 1, 1,
        "IM2COL 2D 3x3 s1 p1 IC=3 8x8 N=4");

    // Non tile-aligned dimensions
    make_im2col_2d_test(5, 13, 17, 7, 3, 3, 2,  1, 1, 1, 1, 1, 1,
        "IM2COL 2D 3x3 s1 p1 IC=5 17x13 N=2 (non tile-aligned)");

    // Dilation > 1
    make_im2col_2d_test(8, 16, 16, 16, 3, 3, 1,  1, 1, 2, 2, 2, 2,
        "IM2COL 2D 3x3 s1 p2 d2 IC=8 16x16 N=1 (dilated)");

    // Patch embedding style (large kernel, stride = kernel size)
    make_im2col_2d_test(3, 32, 32, 768, 16, 16, 1,  16, 16, 0, 0, 1, 1,
        "IM2COL 2D 16x16 s16 p0 IC=3 32x32 N=1 (patch embed)");


    // 1D im2col test
    tests.push_back(make_test([](ggml_context* ctx) {
        const int64_t KW = 3, IC = 8, OC = 16, IW = 32, N = 2;
        // src0 (kernel): ne [KW, IC, OC, 1]
        ggml_tensor* kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KW, IC, OC);
        // src1 (image): ne [IW, IC, N, 1]
        ggml_tensor* image  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, IW, IC, N);
        return ggml_im2col(ctx, kernel, image, 1, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);
    }, "IM2COL 1D KW=3 s1 p1 IC=8 IW=32 N=2", 1e-3));
}

static void add_group_norm_tests(std::vector<std::unique_ptr<test_case>>& tests)
{
    auto make_gn_test = [&](int64_t W, int64_t H, int64_t C, int64_t N,
                            int n_groups, float eps, const char* name) {
        tests.push_back(make_test([=](ggml_context* ctx) {
            // GGML ne: [W, H, C, N]
            ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, N);
            return ggml_group_norm(ctx, input, n_groups, eps);
        }, name, 1e-2));
    };

    // Basic: 32 groups over 32 channels (1 channel/group), tile-aligned spatial
    make_gn_test(32, 32, 32, 1, 32, 1e-6f,
        "GroupNorm 32x32 C=32 G=32 N=1");

    // Typical SD VAE: 32 groups over 128 channels
    make_gn_test(64, 64, 128, 1, 32, 1e-6f,
        "GroupNorm 64x64 C=128 G=32 N=1 (SD VAE)");

    // Smaller spatial, more channels
    make_gn_test(16, 16, 256, 1, 32, 1e-6f,
        "GroupNorm 16x16 C=256 G=32 N=1");

    // Batch > 1
    make_gn_test(32, 32, 64, 2, 32, 1e-6f,
        "GroupNorm 32x32 C=64 G=32 N=2");

    // Non tile-aligned spatial
    make_gn_test(13, 17, 32, 1, 32, 1e-6f,
        "GroupNorm 13x17 C=32 G=32 N=1 (non tile-aligned)");

    // Fewer groups
    make_gn_test(32, 32, 64, 1, 8, 1e-6f,
        "GroupNorm 32x32 C=64 G=8 N=1");

    // Run twice to test program cache reuse (the original crash)
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 32, 32, 1);
        return ggml_group_norm(ctx, input, 32, 1e-6f);
    }, "GroupNorm 32x32 C=32 G=32 N=1 (cache reuse)", 1e-2));
}

static void add_upscale_tests(std::vector<std::unique_ptr<test_case>>& tests)
{
    auto make_upscale_test = [&](int64_t W, int64_t H, int64_t C, int64_t N,
                                  int scale_factor, const char* name) {
        tests.push_back(make_test([=](ggml_context* ctx) {
            ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, N);
            return ggml_upscale(ctx, input, scale_factor, GGML_SCALE_MODE_NEAREST);
        }, name, 1e-3));
    };

    // Basic 2x upscale (most common in SD VAE)
    make_upscale_test(32, 32, 32, 1, 2, "Upscale 2x 32x32 C=32 N=1");
    make_upscale_test(64, 64, 128, 1, 2, "Upscale 2x 64x64 C=128 N=1 (SD VAE)");
    make_upscale_test(16, 16, 256, 1, 2, "Upscale 2x 16x16 C=256 N=1");
    make_upscale_test(32, 32, 64, 2, 2, "Upscale 2x 32x32 C=64 N=2 (batch)");

    // Non-tile-aligned dimensions
    make_upscale_test(13, 17, 32, 1, 2, "Upscale 2x 13x17 C=32 N=1 (non tile-aligned)");

    // 4x upscale
    make_upscale_test(16, 16, 32, 1, 4, "Upscale 4x 16x16 C=32 N=1");

    // Small dimensions
    make_upscale_test(8, 8, 3, 1, 2, "Upscale 2x 8x8 C=3 N=1 (small)");
}

static void add_conv2d_direct_tests(std::vector<std::unique_ptr<test_case>>& tests)
{
    // Helper to create a conv2d direct test
    // kernel: [KW, KH, IC, OC], input: [IW, IH, IC, N]
    auto make_conv2d_test = [&](int64_t IC, int64_t IH, int64_t IW, int64_t OC,
                                int64_t KH, int64_t KW, int64_t N,
                                int s0, int s1, int p0, int p1, int d0, int d1,
                                const char* name, float max_err = 1e-2) {
        tests.push_back(make_test([=](ggml_context* ctx) {
            ggml_tensor* kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, IC, OC);
            ggml_tensor* image  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IW, IH, IC, N);
            return ggml_conv_2d_direct(ctx, kernel, image, s0, s1, p0, p1, d0, d1);
        }, name, max_err));
    };

    // Basic 3x3 conv, stride 1, padding 1
    make_conv2d_test(3, 8, 8, 16, 3, 3, 1,  1, 1, 1, 1, 1, 1,
        "CONV_2D 3x3 s1 p1 IC=3 8x8 N=1");

    // 3x3 conv, stride 1, no padding
    make_conv2d_test(16, 16, 16, 32, 3, 3, 1,  1, 1, 0, 0, 1, 1,
        "CONV_2D 3x3 s1 p0 IC=16 16x16 N=1");

    // 3x3 conv, stride 2, padding 1 (downsampling)
    make_conv2d_test(32, 32, 32, 64, 3, 3, 1,  2, 2, 1, 1, 1, 1,
        "CONV_2D 3x3 s2 p1 IC=32 32x32 N=1");

    // 1x1 conv (pointwise)
    make_conv2d_test(64, 16, 16, 128, 1, 1, 1,  1, 1, 0, 0, 1, 1,
        "CONV_2D 1x1 s1 p0 IC=64 16x16 N=1");

    // Batch > 1
    make_conv2d_test(3, 8, 8, 16, 3, 3, 4,  1, 1, 1, 1, 1, 1,
        "CONV_2D 3x3 s1 p1 IC=3 8x8 N=4");

    // Non tile-aligned dimensions
    make_conv2d_test(5, 13, 17, 7, 3, 3, 2,  1, 1, 1, 1, 1, 1,
        "CONV_2D 3x3 s1 p1 IC=5 17x13 N=2 (non tile-aligned)");

    // Dilation > 1
    make_conv2d_test(8, 16, 16, 16, 3, 3, 1,  1, 1, 2, 2, 2, 2,
        "CONV_2D 3x3 s1 p2 d2 IC=8 16x16 N=1 (dilated)");

    // Patch embedding style (large kernel, stride = kernel size)
    make_conv2d_test(3, 32, 32, 768, 16, 16, 1,  16, 16, 0, 0, 1, 1,
        "CONV_2D 16x16 s16 p0 IC=3 32x32 N=1 (patch embed)");

    // SD VAE typical: 3x3 s1 p1 over 128-channel 64x64
    make_conv2d_test(128, 64, 64, 128, 3, 3, 1,  1, 1, 1, 1, 1, 1,
        "CONV_2D 3x3 s1 p1 IC=128 64x64 N=1 (SD VAE)");

    // Small spatial, many channels (bottleneck)
    make_conv2d_test(256, 8, 8, 256, 3, 3, 1,  1, 1, 1, 1, 1, 1,
        "CONV_2D 3x3 s1 p1 IC=256 8x8 N=1 (bottleneck)");
}

// Hypothesis #1 diagnostic: VAE 1x1-conv channel<->width layout scramble.
//
// The 1x1 fast path in ggml_backend_metalium_conv2d_direct (ggml-metalium.cpp,
// the `if (KH==1 && KW==1 && s0==1 && s1==1 && p0==0 && p1==0)` branch) reshapes
// the NCHW input [N,IC,IH,IW] straight to [N*IH*IW, IC] WITHOUT first permuting
// to NHWC. The general (>=2x2 / padded) path right below it DOES permute to NHWC
// before flattening. If that permute is missing, every 1x1 conv multiplies the
// weight against width-neighbours instead of channels -> a whole-tensor scramble,
// while >=2x2 convs stay correct.
//
// Each SUSPECT (1x1) case below is paired with a CONTROL (3x3) case on the SAME
// IC/IH/IW/OC/N. Interpretation when running test-metalium:
//   * SUSPECT fails (large nmse) + CONTROL passes  => hypothesis #1 CONFIRMED,
//     and localized precisely to the 1x1 fast path.
//   * Both pass                                    => #1 is NOT the cause; the
//     garbage originates elsewhere (re-run without --diffusion-fa, or dump the
//     pre-VAE latent and compare CPU-vs-Metalium).
//   * Both fail                                    => conv2d_direct is broken for
//     more than just the 1x1 path; widen the search.
static void add_conv2d_1x1_diag_tests(std::vector<std::unique_ptr<test_case>>& tests)
{
    // s=1; p=0 with KH=KW=1 hits the 1x1 fast path, p=1 with KH=KW=3 hits the
    // general NHWC-permute path. nmse threshold 1e-2 cleanly separates a correct
    // matmul (<1e-2) from a channel<->width scramble (~O(0.1..2)).
    auto conv = [&](int64_t IC, int64_t IH, int64_t IW, int64_t OC, int64_t N,
                    int64_t KH, int64_t KW, int p, const char* name) {
        tests.push_back(make_test([=](ggml_context* ctx) {
            ggml_tensor* kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, IC, OC);
            ggml_tensor* image  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IW, IH, IC, N);
            return ggml_conv_2d_direct(ctx, kernel, image, 1, 1, p, p, 1, 1);
        }, name, 1e-2));
    };

    // Matched pair: identical shape, only the kernel size (and thus dispatch
    // path) differs. This is the decisive A/B test for #1.
    conv(64, 16, 16, 128, 1, 3, 3, 1, "[#1 CONTROL] 3x3 s1 p1 IC=64 16x16 OC=128 N=1");
    conv(64, 16, 16, 128, 1, 1, 1, 0, "[#1 SUSPECT] 1x1 s1 p0 IC=64 16x16 OC=128 N=1");

    // Asymmetric IC != IH != IW so a channel<->width confusion cannot accidentally
    // coincide with the correct answer.
    conv(8, 4, 16, 8, 1, 3, 3, 1, "[#1 CONTROL] 3x3 IC=8 IH=4 IW=16 OC=8 N=1 (asym)");
    conv(8, 4, 16, 8, 1, 1, 1, 0, "[#1 SUSPECT] 1x1 IC=8 IH=4 IW=16 OC=8 N=1 (asym)");

    // Representative z-image VAE 1x1 sites (the convs that actually run in decode).
    conv(16,  32, 32, 16,  1, 1, 1, 0, "[#1 SUSPECT] 1x1 post_quant_conv-like IC=16 32x32 OC=16");
    conv(512, 32, 32, 512, 1, 1, 1, 0, "[#1 SUSPECT] 1x1 mid.attn q/k/v/proj-like IC=512 32x32 OC=512");
    conv(256, 64, 64, 512, 1, 1, 1, 0, "[#1 SUSPECT] 1x1 nin_shortcut-like IC=256->512 64x64");

    // Tiny case, easy to reason about by hand if you dump the buffers.
    conv(2, 2, 4, 2, 1, 1, 1, 0, "[#1 SUSPECT] 1x1 IC=2 IH=2 IW=4 OC=2 N=1 (tiny)");
}

int main(int argc, char ** argv)
{
    (void)argc;
    (void)argv;
    ggml_backend_t cpu = ggml_backend_cpu_init();

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Metalium");
    if(reg == NULL) {
        fprintf(stderr, "Cannot find the Metalium backend. Is the Meralium backend disabled?\n");
        return 1;
    }
    if(ggml_backend_reg_dev_count(reg) == 0) {
        fprintf(stderr, "No devices found for Metalium backend. Is the kernel driver working?\n");
        return 1;
    }
    ggml_backend_t metalium = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), NULL);

    std::vector<std::unique_ptr<test_case>> tests;
    add_conv2d_1x1_diag_tests(tests);  // hypothesis #1: run first so an unrelated crash later doesn't mask it
    add_unittests(tests);
    add_flux_rope_tests(tests);
    add_apply_rope_full_path_tests(tests);
    add_lowprec_matmul_tests(tests);
    add_inplace_view_tests(tests);
    add_im2col_tests(tests);
    add_group_norm_tests(tests);
    add_upscale_tests(tests);
    add_conv2d_direct_tests(tests);

    ///////////////// put experiment code here /////////////////
    // easier on the eye to find it (also one line to disable UT)
    // RMS_NORM: stage bisection of z-image diffusion pinpointed cap_embedder's
    // RMSNorm as the broken op (Metalium output ~0 vs CPU large). No prior test.
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        return ggml_rms_norm(ctx, a, 1e-6f);
    }, "RMSNorm 64x64 basic", 1e-3));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 122);
        return ggml_rms_norm(ctx, a, 1e-6f);
    }, "RMSNorm 2560x122 (cap_embedder shape)", 1e-3));
    // Large-magnitude / outlier input (mimics Qwen massive activations) by scaling
    // half the rows up; RMSNorm is per-row so this stresses the variance reduction.
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 16);
        ggml_tensor* s = ggml_scale(ctx, a, 1000.0f);
        return ggml_rms_norm(ctx, s, 1e-6f);
    }, "RMSNorm 2560x16 large-magnitude", 1e-3));
    // Broadcast MUL (vector * matrix) — the RMSNorm weight-apply pattern.
    // RMSNorm does ggml_mul(x[hidden,tokens], w[hidden]); suspected broken on Metalium.
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 64);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
        return ggml_mul(ctx, a, b);
    }, "MUL broadcast vector*matrix 2048x64", 1e-3));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 122);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560);
        return ggml_mul(ctx, a, b);
    }, "MUL broadcast vector*matrix 2560x122 (cap_embedder)", 1e-3));
    // Large-magnitude weight (mimics the ~1000-magnitude RMSNorm weights on outlier channels).
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 122);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560);
        return ggml_mul(ctx, a, ggml_scale(ctx, b, 1000.0f));
    }, "MUL broadcast large-weight 2560x122", 1e-2));
    // The actual RMSNorm class pattern: rms_norm then broadcast mul.
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 122);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560);
        return ggml_mul(ctx, ggml_rms_norm(ctx, a, 1e-6f), b);
    }, "RMSNorm+weight 2560x122 (full cap pattern)", 1e-3));
    // MASSIVE-ACTIVATION pattern: a few channels ~6000x larger than the rest,
    // mimicking the Qwen text-encoder outliers that feed cap_embedder. RMSNorm
    // is NOT scale-invariant under this (extreme per-row dynamic range) — this is
    // the case the stage-bisection implicated.
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* spike  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 16);    // outlier channels
        ggml_tensor* normal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2528, 16);  // normal channels
        ggml_tensor* x = ggml_concat(ctx, ggml_scale(ctx, spike, 6000.0f), normal, 0);  // [2560,16]
        ggml_tensor* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560);
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
    }, "RMSNorm+weight massive-activation outliers 2560x16", 1e-2));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* spike  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 16);
        ggml_tensor* normal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2528, 16);
        ggml_tensor* x = ggml_concat(ctx, ggml_scale(ctx, spike, 6000.0f), normal, 0);
        return ggml_rms_norm(ctx, x, 1e-6f);  // just the op, no weight
    }, "RMSNorm massive-activation outliers 2560x16 (op only)", 1e-2));
    // Real token count 122 (NON tile-aligned, pads to 128) WITH outliers — the
    // exact cap_embedder case. Non-tile-aligned token dim + extreme dynamic range.
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* spike  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 122);
        ggml_tensor* normal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2528, 122);
        ggml_tensor* x = ggml_concat(ctx, ggml_scale(ctx, spike, 6000.0f), normal, 0);  // [2560,122]
        ggml_tensor* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560);
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
    }, "RMSNorm+weight outliers 2560x122 (real cap case)", 1e-2));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* spike  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 122);
        ggml_tensor* normal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2528, 122);
        ggml_tensor* x = ggml_concat(ctx, ggml_scale(ctx, spike, 6000.0f), normal, 0);
        return ggml_rms_norm(ctx, x, 1e-6f);  // op only
    }, "RMSNorm outliers 2560x122 (op only)", 1e-2));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 32, 1, 1);
        ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 32, 1, 1);
        return ggml_soft_max_ext(ctx, a, mask, 1, 0);
    }, "test softmax 0", 1e-5));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1023, 31, 1, 1);
        ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1023, 31, 1, 1);
        return ggml_soft_max_ext(ctx, a, mask, 1, 0);
    }, "test softmax 1", 1e-5));
    ///////////////// end of experiment code /////////////////

    size_t total_tests = 0;
    size_t passed_tests = 0;
    size_t not_supported = 0;
    for(auto& test : tests) {
        TestResult res = test->eval(metalium, cpu);

        total_tests++;
        if (res == TestResult::OK) {
            passed_tests++;
        } else if (res == TestResult::NOT_SUPPORTED) {
            not_supported++;
        }
    }

    double passed_ratio = (double)passed_tests / total_tests;
    double not_supported_ratio = (double)not_supported / total_tests;
    std::cout << "\nStats for Metalium backend: " << (total_tests != passed_tests ? "\033[1;31mFAIL\033[0m\n" : "\033[1;32mOK\033[0m\n")
        << "  Test status: " << passed_tests << " / " << total_tests << " passed\n"
        << "  Failed: " << total_tests - passed_tests - not_supported << " (" << std::round(1.0 - passed_ratio - not_supported_ratio) * 100 << "%)\n"
        << "  Not supported: " << not_supported << "\n";

    bool failed = total_tests != passed_tests;
    if(failed) {
        std::cout << "Some tests failed\n";
    }

    ggml_backend_free(metalium);
    ggml_backend_free(cpu);

    return (int)failed;
}

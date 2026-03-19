#include "kernels/gpu.h"
#include "kernels/silu/kernel.h"
#include "model.h"

#define SILU_UNIFORM_WORDS 3u

enum { SILU_MAX_PADDED = (((MAX_SEQ_LEN * INTERMEDIATE_SIZE) + 191u) / 192u) * 192u };

static void silu_build_uniforms(uint32_t uniforms[GPU_NUM_QPUS][SILU_UNIFORM_WORDS],
                                uint32_t uniform_ptrs[GPU_NUM_QPUS],
                                float *input,
                                float *output,
                                unsigned num_elements) {
    unsigned total_groups = num_elements / 16;
    unsigned groups_per_qpu = total_groups / GPU_NUM_QPUS;

    for (unsigned q = 0; q < GPU_NUM_QPUS; q++) {
        uint32_t *u = uniforms[q];
        unsigned start_group = q * groups_per_qpu;
        u[0] = groups_per_qpu;
        u[1] = gpu_bus_addr(input) + start_group * 64u;
        u[2] = gpu_bus_addr(output) + start_group * 64u;
        uniform_ptrs[q] = gpu_bus_addr(u);
    }
}

void silu_kernel_inplace(Matrix x) {
    unsigned used = (unsigned)num_elements(x);
    unsigned padded = gpu_pad_qpu(used);

    float input[SILU_MAX_PADDED] __attribute__((aligned(16)));
    float output[SILU_MAX_PADDED] __attribute__((aligned(16)));
    uint32_t uniforms[GPU_NUM_QPUS][SILU_UNIFORM_WORDS] __attribute__((aligned(16)));
    uint32_t uniform_ptrs[GPU_NUM_QPUS];

    gpu_copy_matrix(x, input);
    for (unsigned i = used; i < padded; i++) {
        input[i] = 0.0f;
    }

    silu_build_uniforms(uniforms, uniform_ptrs, input, output, padded);
    gpu_exec_direct(gpu_bus_addr(silu_raw_kernel), uniform_ptrs, GPU_NUM_QPUS);
    gpu_copy_array_into_matrix(output, x);
}

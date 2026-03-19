#include "kernels/gating/kernel.h"
#include "kernels/gpu.h"
#include "model.h"

#define GATING_UNIFORM_WORDS 4u

enum { GATING_MAX_PADDED = (((MAX_SEQ_LEN * INTERMEDIATE_SIZE) + 191u) / 192u) * 192u };

static void gating_build_uniforms(uint32_t uniforms[GPU_NUM_QPUS][GATING_UNIFORM_WORDS],
                                  uint32_t uniform_ptrs[GPU_NUM_QPUS],
                                  float *x,
                                  float *y,
                                  float *out,
                                  unsigned num_elements) {
    unsigned total_groups = num_elements / 16;
    unsigned groups_per_qpu = total_groups / GPU_NUM_QPUS;

    for (unsigned q = 0; q < GPU_NUM_QPUS; q++) {
        uint32_t *u = uniforms[q];
        unsigned start_group = q * groups_per_qpu;
        u[0] = groups_per_qpu;
        u[1] = gpu_bus_addr(x) + start_group * 64u;
        u[2] = gpu_bus_addr(y) + start_group * 64u;
        u[3] = gpu_bus_addr(out) + start_group * 64u;
        uniform_ptrs[q] = gpu_bus_addr(u);
    }
}

void gating_kernel_into(Matrix a, Matrix b, Matrix out) {
    unsigned used = (unsigned)num_elements(a);
    unsigned padded = gpu_pad_qpu(used);

    float x_data[GATING_MAX_PADDED] __attribute__((aligned(16)));
    float y_data[GATING_MAX_PADDED] __attribute__((aligned(16)));
    float out_data[GATING_MAX_PADDED] __attribute__((aligned(16)));
    uint32_t uniforms[GPU_NUM_QPUS][GATING_UNIFORM_WORDS] __attribute__((aligned(16)));
    uint32_t uniform_ptrs[GPU_NUM_QPUS];

    gpu_copy_matrix(a, x_data);
    gpu_copy_matrix(b, y_data);
    for (unsigned i = used; i < padded; i++) {
        x_data[i] = 0.0f;
        y_data[i] = 0.0f;
    }

    gating_build_uniforms(uniforms, uniform_ptrs, x_data, y_data, out_data, padded);
    gpu_exec_direct(gpu_bus_addr(gating_raw_kernel), uniform_ptrs, GPU_NUM_QPUS);
    gpu_copy_array_into_matrix(out_data, out);
}

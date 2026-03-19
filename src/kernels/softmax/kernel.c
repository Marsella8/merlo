#include <math.h>
#include <string.h>

#include "kernels/gpu.h"
#include "kernels/softmax/kernel.h"
#include "model.h"

#define SOFTMAX_UNIFORM_WORDS 4u

enum { SOFTMAX_IO_FLOATS = MAX_SEQ_LEN * 256 };

void softmax_kernel_into(Matrix x, Matrix out) {
    float input[SOFTMAX_IO_FLOATS] __attribute__((aligned(16)));
    float output[SOFTMAX_IO_FLOATS] __attribute__((aligned(16)));
    uint32_t uniforms[GPU_NUM_QPUS][SOFTMAX_UNIFORM_WORDS] __attribute__((aligned(16)));
    uint32_t uniform_ptrs[GPU_NUM_QPUS];

    unsigned rows = (unsigned)x.rows;
    unsigned cols = (unsigned)x.cols;
    unsigned stride = gpu_pad16(cols);

    unsigned active = rows < GPU_NUM_QPUS ? rows : GPU_NUM_QPUS;
    unsigned base_count = rows / active;
    unsigned extra_count = rows % active;
    unsigned start = 0;

    gpu_copy_matrix_rows_padded(x, input, stride, 0.0f);
    memset(output, 0, (size_t)rows * (size_t)stride * sizeof(float));

    for (unsigned q = 0; q < GPU_NUM_QPUS; q++) {
        uint32_t *u = uniforms[q];
        if (q < active) {
            unsigned count = base_count + (q < extra_count ? 1u : 0u);
            u[0] = count;
            u[1] = stride;
            u[2] = gpu_bus_addr(input) + start * stride * sizeof(float);
            u[3] = gpu_bus_addr(output) + start * stride * sizeof(float);
            start += count;
        } else {
            u[0] = 0;
            u[1] = stride;
            u[2] = 0;
            u[3] = 0;
        }
        uniform_ptrs[q] = gpu_bus_addr(u);
    }

    for (unsigned r = 0; r < rows; r++) {
        for (unsigned c = cols; c < stride; c++)
            input[r * stride + c] = -INFINITY;
    }

    gpu_exec_direct(gpu_bus_addr(softmax_raw_kernel), uniform_ptrs, GPU_NUM_QPUS);
    gpu_copy_padded_rows_into_matrix(output, stride, out);
}

#include <string.h>

#include "kernels/gpu.h"
#include "kernels/q4_0_matvec/kernel.h"
#include "model.h"

#define Q4_ROWS_PER_PANEL 16u
#define Q4_COLS_PER_BLOCK 32u
#define Q4_BLOCK_BYTES 320u
#define Q4_0_MATVEC_MAX_INPUT_COLS INTERMEDIATE_SIZE
#define Q4_0_MATVEC_MAX_OUTPUT_ROWS VOCAB_SIZE
#define Q4_0_MATVEC_INPUT_WORDS (33u * (Q4_0_MATVEC_MAX_INPUT_COLS / Q4_COLS_PER_BLOCK))

static float q4_0_matvec_output[Q4_0_MATVEC_MAX_OUTPUT_ROWS] __attribute__((aligned(16)));
static uint32_t q4_0_matvec_input[Q4_0_MATVEC_INPUT_WORDS] __attribute__((aligned(16)));
static uint32_t q4_0_matvec_uniforms[GPU_NUM_QPUS][6] __attribute__((aligned(16)));
static uint32_t q4_0_matvec_uniform_ptrs[GPU_NUM_QPUS] __attribute__((aligned(16)));

static void q4_0_matvec_build_uniforms(Matrix x, QMatrix w) {
    size_t num_blocks = w.cols / Q4_COLS_PER_BLOCK;
    size_t num_panels = w.rows / Q4_ROWS_PER_PANEL;
    size_t panels_per_qpu = (num_panels + GPU_NUM_QPUS - 1) / GPU_NUM_QPUS;
    size_t panel_stride = num_blocks * Q4_BLOCK_BYTES;

    size_t idx = 0;
    for (size_t block = 0; block < num_blocks; block++) {
        float sum_x = 0.0f;
        for (size_t i = 0; i < Q4_COLS_PER_BLOCK; i++) {
            float xv = *at(x, 0, block * Q4_COLS_PER_BLOCK + i);
            q4_0_matvec_input[idx++] = gpu_bits_from_float(xv);
            sum_x += xv;
        }
        q4_0_matvec_input[idx++] = gpu_bits_from_float(8.0f * sum_x);
    }

    for (size_t q = 0; q < GPU_NUM_QPUS; q++) {
        uint32_t *u = q4_0_matvec_uniforms[q];
        size_t start = q * panels_per_qpu;
        size_t end = start + panels_per_qpu;
        if (end > num_panels)
            end = num_panels;

        u[0] = (uint32_t)(end - start);
        u[1] = (uint32_t)num_blocks;
        u[2] = gpu_bus_addr((const uint8_t *)w.buffer->data + start * panel_stride);
        u[3] = gpu_bus_addr(q4_0_matvec_output + start * Q4_ROWS_PER_PANEL);
        u[4] = (uint32_t)panel_stride;
        u[5] = gpu_bus_addr(q4_0_matvec_input);

        q4_0_matvec_uniform_ptrs[q] = gpu_bus_addr(u);
    }
}

void q4_0_matvec_into(Matrix x, QMatrix w, Matrix out) {
    memset(q4_0_matvec_output, 0, w.rows * sizeof(float));
    q4_0_matvec_build_uniforms(x, w);
    gpu_exec_direct(gpu_bus_addr(q4_0_matvec_raw_kernel), q4_0_matvec_uniform_ptrs, GPU_NUM_QPUS);

    for (size_t j = 0; j < w.rows; j++)
        *at(out, 0, j) = q4_0_matvec_output[j];
}

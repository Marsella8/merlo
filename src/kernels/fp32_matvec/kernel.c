#include <string.h>

#include "kernels/fp32_matvec/kernel.h"
#include "kernels/gpu.h"
#include "model.h"

#define FP32_MATVEC_UNIFORM_BASE_WORDS 6u
#define FP32_UNIFORM_STRIDE (FP32_MATVEC_UNIFORM_BASE_WORDS + INTERMEDIATE_SIZE)

enum {
    FP32_MATVEC_ROWS_PAD_MAX = 384u,
    FP32_MATVEC_PANEL_FLOATS = 49152u,
};

static float fp32_row_scratch[INTERMEDIATE_SIZE] __attribute__((aligned(16)));

void fp32_matvec_into(Matrix x, Matrix w, Matrix out) {
    float flat[FP32_MATVEC_PANEL_FLOATS] __attribute__((aligned(16)));
    float panel[FP32_MATVEC_PANEL_FLOATS] __attribute__((aligned(16)));
    float output[FP32_MATVEC_ROWS_PAD_MAX] __attribute__((aligned(16)));
    uint32_t uniforms[GPU_NUM_QPUS][FP32_UNIFORM_STRIDE] __attribute__((aligned(16)));
    uint32_t uniform_ptrs[GPU_NUM_QPUS];

    unsigned rows = (unsigned)w.rows;
    unsigned cols = (unsigned)w.cols;
    unsigned rows_pad = gpu_pad_qpu(rows);
    unsigned num_panels = rows_pad / 16u;
    unsigned panels_per_qpu = (num_panels + GPU_NUM_QPUS - 1u) / GPU_NUM_QPUS;
    unsigned panel_stride = cols * 64u;
    size_t panel_elems = (size_t)rows_pad * (size_t)cols;

    if (x.row_stride > 0 && x.col_stride == 1) {
        const float *src = &((float *)x.buffer->data)[x.offset];
        memcpy(fp32_row_scratch, src, cols * sizeof(float));
    } else {
        for (unsigned k = 0; k < cols; k++)
            fp32_row_scratch[k] = *at_f32(x, 0, k);
    }

    memset(flat, 0, panel_elems * sizeof(float));
    gpu_copy_matrix_rows_padded(w, flat, cols, 0.0f);
    gpu_fp32_to_panel_layout(flat, panel, rows_pad, cols);
    memset(output, 0, (size_t)rows_pad * sizeof(float));

    for (unsigned q = 0; q < GPU_NUM_QPUS; q++) {
        uint32_t *u = uniforms[q];
        unsigned start = q * panels_per_qpu;
        unsigned end = start + panels_per_qpu;
        if (end > num_panels)
            end = num_panels;
        u[0] = end - start;
        u[1] = cols;
        u[2] = gpu_bus_addr(panel) + start * panel_stride;
        u[3] = gpu_bus_addr(output) + start * 16u * sizeof(float);
        u[4] = panel_stride;
        u[5] = gpu_bus_addr(&u[6]);
        memcpy(&u[6], fp32_row_scratch, cols * sizeof(float));
        uniform_ptrs[q] = gpu_bus_addr(u);
    }

    gpu_exec_direct(gpu_bus_addr(fp32_matvec_raw_kernel), uniform_ptrs, GPU_NUM_QPUS);
    for (unsigned i = 0; i < rows; i++)
        *at(out, 0, i) = output[i];
}

void fp32_matmul_into(Matrix a, Matrix b, Matrix out) {
    enum {
        k_rows_pad = (((HEAD_DIM) + 191u) / 192u) * 192u,
        k_cols = MAX_SEQ_LEN,
        k_panel = k_rows_pad * k_cols,
    };

    float flat[k_panel] __attribute__((aligned(16)));
    float panel[k_panel] __attribute__((aligned(16)));
    float output[k_rows_pad] __attribute__((aligned(16)));
    uint32_t uniforms[GPU_NUM_QPUS][FP32_UNIFORM_STRIDE] __attribute__((aligned(16)));
    uint32_t uniform_ptrs[GPU_NUM_QPUS];

    Matrix b_t = transpose(b);
    unsigned rows = (unsigned)b_t.rows;
    unsigned cols = (unsigned)b_t.cols;
    unsigned rows_pad = gpu_pad_qpu(rows);
    unsigned num_panels = rows_pad / 16u;
    unsigned panels_per_qpu = (num_panels + GPU_NUM_QPUS - 1u) / GPU_NUM_QPUS;
    unsigned panel_stride = cols * 64u;

    memset(flat, 0, (size_t)rows_pad * (size_t)cols * sizeof(float));
    gpu_copy_matrix_rows_padded(b_t, flat, cols, 0.0f);
    gpu_fp32_to_panel_layout(flat, panel, rows_pad, cols);

    for (unsigned i = 0; i < (unsigned)a.rows; i++) {
        if (a.row_stride > 0 && a.col_stride == 1) {
            const float *src =
                &((float *)a.buffer->data)[a.offset + (size_t)i * (size_t)a.row_stride];
            memcpy(fp32_row_scratch, src, cols * sizeof(float));
        } else {
            for (unsigned k = 0; k < cols; k++)
                fp32_row_scratch[k] = *at_f32(a, i, k);
        }

        memset(output, 0, (size_t)rows_pad * sizeof(float));

        for (unsigned q = 0; q < GPU_NUM_QPUS; q++) {
            uint32_t *u = uniforms[q];
            unsigned start = q * panels_per_qpu;
            unsigned end = start + panels_per_qpu;
            if (end > num_panels)
                end = num_panels;
            u[0] = end - start;
            u[1] = cols;
            u[2] = gpu_bus_addr(panel) + start * panel_stride;
            u[3] = gpu_bus_addr(output) + start * 16u * sizeof(float);
            u[4] = panel_stride;
            u[5] = gpu_bus_addr(&u[6]);
            memcpy(&u[6], fp32_row_scratch, cols * sizeof(float));
            uniform_ptrs[q] = gpu_bus_addr(u);
        }

        gpu_exec_direct(gpu_bus_addr(fp32_matvec_raw_kernel), uniform_ptrs, GPU_NUM_QPUS);

        for (unsigned j = 0; j < rows; j++)
            *at(out, i, j) = output[j];
    }
}

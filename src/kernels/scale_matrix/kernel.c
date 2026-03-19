#include <string.h>

#include "kernels/gpu.h"
#include "kernels/scale_matrix/kernel.h"
#include "model.h"

void scale_matrix_into(Matrix x, float alpha, Matrix out) {
    enum { k_panel = 384 * MAX_SEQ_LEN };

    float flat_in[k_panel] __attribute__((aligned(16)));
    float flat_out[k_panel] __attribute__((aligned(16)));
    float panel_in[k_panel] __attribute__((aligned(16)));
    float panel_out[k_panel] __attribute__((aligned(16)));
    uint32_t uniforms[GPU_NUM_QPUS][SCALE_MATRIX_UNIFORM_WORDS] __attribute__((aligned(16)));
    uint32_t uniform_ptrs[GPU_NUM_QPUS];

    unsigned rows = (unsigned)x.rows;
    unsigned cols = (unsigned)x.cols;
    unsigned rows_pad = gpu_pad_qpu(rows);
    size_t panel_elems = (size_t)rows_pad * (size_t)cols;

    unsigned num_panels = rows_pad / 16u;
    unsigned panels_per_qpu = (num_panels + GPU_NUM_QPUS - 1u) / GPU_NUM_QPUS;
    unsigned panel_stride = cols * 64u;
    uint32_t alpha_bits = gpu_bits_from_float(alpha);

    memset(flat_in, 0, panel_elems * sizeof(float));
    gpu_copy_matrix_rows_padded(x, flat_in, cols, 0.0f);
    gpu_fp32_to_panel_layout(flat_in, panel_in, rows_pad, cols);
    memset(panel_out, 0, panel_elems * sizeof(float));

    for (unsigned q = 0; q < GPU_NUM_QPUS; q++) {
        uint32_t *u = uniforms[q];
        unsigned start = q * panels_per_qpu;
        unsigned end = start + panels_per_qpu;
        if (end > num_panels)
            end = num_panels;
        u[0] = end - start;
        u[1] = cols;
        u[2] = gpu_bus_addr(panel_in) + start * panel_stride;
        u[3] = gpu_bus_addr(panel_out) + start * panel_stride;
        u[4] = panel_stride;
        u[5] = alpha_bits;
        uniform_ptrs[q] = gpu_bus_addr(u);
    }

    gpu_exec_direct(gpu_bus_addr(scale_matrix_raw_kernel), uniform_ptrs, GPU_NUM_QPUS);
    gpu_fp32_from_panel_layout(panel_out, flat_out, rows_pad, cols);
    gpu_copy_padded_rows_into_matrix(flat_out, cols, out);
}

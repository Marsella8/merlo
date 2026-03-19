#ifndef KERNELS_GPU_H
#define KERNELS_GPU_H

#include <stdint.h>

#include "matrix.h"

#define GPU_BUS_ALIAS 0x40000000u
#define GPU_NUM_QPUS 12u

void gpu_init(void);
void gpu_exec_direct(uint32_t code, uint32_t uniform_ptrs[], unsigned num_qpus);
uint32_t gpu_bus_addr(const void *ptr);
uint32_t gpu_bits_from_float(float value);
uint16_t gpu_fp32_to_bf16(float value);
unsigned gpu_pad16(unsigned n);
unsigned gpu_pad_qpu(unsigned n);
float *gpu_matrix_data(Matrix x);
void gpu_copy_matrix(Matrix src, float *dst);
void gpu_copy_array_into_matrix(const float *src, Matrix dst);
void gpu_copy_matrix_rows_padded(Matrix src, float *dst, unsigned stride, float pad_value);
void gpu_copy_padded_rows_into_matrix(const float *src, unsigned stride, Matrix dst);
void gpu_fp32_to_panel_layout(const float *src, float *panel, unsigned rows, unsigned cols);
void gpu_fp32_from_panel_layout(const float *panel, float *dst, unsigned rows, unsigned cols);
void gpu_bf16_to_panel_layout(const uint16_t *src, uint16_t *panel, unsigned rows, unsigned cols);

#endif

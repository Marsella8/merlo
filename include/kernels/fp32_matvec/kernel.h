#ifndef KERNELS_FP32_MATVEC_KERNEL_H
#define KERNELS_FP32_MATVEC_KERNEL_H

#include <stdint.h>

#include "matrix.h"

#define FP32_MATVEC_RAW_KERNEL_WORDS 126u

extern uint32_t fp32_matvec_raw_kernel[FP32_MATVEC_RAW_KERNEL_WORDS];

void fp32_matvec_into(Matrix x, Matrix w, Matrix out);
void fp32_matmul_into(Matrix a, Matrix b, Matrix out);

#endif

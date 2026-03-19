#ifndef KERNELS_GATING_KERNEL_H
#define KERNELS_GATING_KERNEL_H

#include <stdint.h>

#include "matrix.h"

#define GATING_RAW_KERNEL_WORDS 114u

extern uint32_t gating_raw_kernel[GATING_RAW_KERNEL_WORDS];

void gating_kernel_into(Matrix a, Matrix b, Matrix out);

#endif

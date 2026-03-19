#ifndef KERNELS_SILU_KERNEL_H
#define KERNELS_SILU_KERNEL_H

#include <stdint.h>

#include "matrix.h"

#define SILU_RAW_KERNEL_WORDS 122u

extern uint32_t silu_raw_kernel[SILU_RAW_KERNEL_WORDS];

void silu_kernel_inplace(Matrix x);

#endif

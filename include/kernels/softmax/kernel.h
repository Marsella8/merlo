#ifndef KERNELS_SOFTMAX_KERNEL_H
#define KERNELS_SOFTMAX_KERNEL_H

#include <stdint.h>

#include "matrix.h"

#define SOFTMAX_RAW_KERNEL_WORDS 338u

extern uint32_t softmax_raw_kernel[SOFTMAX_RAW_KERNEL_WORDS];

void softmax_kernel_into(Matrix x, Matrix out);

#endif

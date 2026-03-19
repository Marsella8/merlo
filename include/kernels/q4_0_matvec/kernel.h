#ifndef KERNELS_Q4_0_MATVEC_KERNEL_H
#define KERNELS_Q4_0_MATVEC_KERNEL_H

#include <stdint.h>

#include "matrix.h"

#define Q4_0_MATVEC_RAW_KERNEL_WORDS 398u

extern uint32_t q4_0_matvec_raw_kernel[Q4_0_MATVEC_RAW_KERNEL_WORDS];

void q4_0_matvec_into(Matrix x, QMatrix w, Matrix out);

#endif

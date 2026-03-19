#ifndef KERNELS_SCALE_MATRIX_KERNEL_H
#define KERNELS_SCALE_MATRIX_KERNEL_H

#include <stdint.h>

#include "kernels/gpu.h"
#include "matrix.h"
#include "model.h"

#define SCALE_MATRIX_RAW_KERNEL_WORDS 124u
#define SCALE_MATRIX_UNIFORM_WORDS 6u

#define SCALE_MATRIX_MAX_PANEL_FLOATS (384u * (unsigned)MAX_SEQ_LEN)

extern uint32_t scale_matrix_raw_kernel[SCALE_MATRIX_RAW_KERNEL_WORDS];

void scale_matrix_into(Matrix x, float alpha, Matrix out);

#endif

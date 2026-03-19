#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "rpi.h"

// basically a fat pointer.
typedef struct {
    void* data; 
    size_t size; // in bytes
    bool owned;
} Buffer;

typedef struct {
    Buffer* buffer; //fp32
    size_t rows;
    size_t cols;
    int row_stride;
    int col_stride;
    size_t offset;
    bool owned;
} Matrix;


// unlike Matrix, these are just passed to kernels so we don't really need the view repr stuff
typedef struct {
    Buffer* buffer;
    size_t rows;
    size_t cols;
} QMatrix;

float* at_f32(Matrix mat, size_t r, size_t c);

#define at(m, r, c) at_f32((m), (r), (c))

void qmatrix_row_decode_into(QMatrix q, size_t row, Matrix dst); // for embedding lookup

Buffer* buf(size_t size);
Buffer* watch_buf(void *data, size_t size);
Matrix stack_mat(Buffer *buffer, float *data, size_t rows, size_t cols);
void free_buf(Buffer* b_ptr);
void free_mat(Matrix m);
void free_qmat(QMatrix q);
Matrix mat(const Buffer* buffer, size_t rows, size_t cols);
Matrix empty(size_t rows, size_t cols);
Matrix as_contiguous(Matrix mat);
Matrix mat_from_array(size_t rows, size_t cols, float m[rows][cols]);
Matrix transpose(Matrix mat);
Matrix slice(Matrix mat, size_t r_start, size_t r_end, size_t c_start, size_t c_end);
Matrix add(Matrix a, Matrix b);
void add_into(Matrix a, Matrix b, Matrix out);
Matrix scale(Matrix a, float value);
void scale_into(Matrix a, float value, Matrix out);
void scale_inplace(Matrix a, float value);
Matrix matmul(Matrix a, Matrix b);
void matmul_into(Matrix a, Matrix b, Matrix out);
Matrix masked_matmul(Matrix a, Matrix b);
Matrix qmatmul(Matrix a, QMatrix b);
void qmatmul_into(Matrix a, QMatrix b, Matrix out);
void matvec_into(Matrix x, Matrix w, Matrix out);
void qmatvec_into(Matrix x, QMatrix b, Matrix out);
size_t qnum_bytes_for_shape(size_t rows, size_t cols);
size_t num_elements(Matrix m);
size_t num_bytes(Matrix m);
bool eq(Matrix a, Matrix b);
void copy(Matrix src, Matrix dst);
Matrix dequantize(QMatrix q);
QMatrix qmat(const Buffer* buffer, size_t rows, size_t cols);

#define assume_shape(m, r, c) assert(((int)(r) == -1 || (m).rows == (size_t)(r)) && ((int)(c) == -1 || (m).cols == (size_t)(c)))

#endif // MATRIX_H

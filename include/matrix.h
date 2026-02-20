#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>

typedef struct {
    void* data; 
    size_t size; // in bytes
} Buffer;

typedef struct {
    Buffer* buffer; //fp32
    size_t rows;
    size_t cols;
    int row_stride;
    int col_stride;
    size_t offset;
} Matrix;

typedef struct {
    Buffer* weights; // int8
    Buffer* scales;  // float, one per block
    size_t rows;
    size_t cols;
    int row_stride;
    int col_stride;
    size_t offset;
} QMatrix;

float* at_f32(Matrix mat, size_t r, size_t c);
int8_t* at_i8(QMatrix mat, size_t r, size_t c);

#define at(m, r, c) _Generic((m), \
    Matrix: at_f32, \
    QMatrix: at_i8  \
)(m, r, c)  

Buffer* buf(size_t size);
void free_buf(Buffer* b_ptr);
void free_mat(Matrix m);
Matrix mat(const Buffer* buffer, size_t rows, size_t cols);
Matrix empty(size_t rows, size_t cols);
Matrix mat_from_array(size_t rows, size_t cols, float m[rows][cols]);
Matrix transpose(Matrix mat);
Matrix slice(Matrix mat, size_t r_start, size_t r_end, size_t c_start, size_t c_end);
Matrix add(Matrix a, Matrix b);
Matrix matmul(Matrix a, Matrix b);
Matrix masked_matmul(Matrix a, Matrix b);
Matrix qmatmul(Matrix a, QMatrix b);
bool eq(Matrix a, Matrix b);
void copy(Matrix src, Matrix dst);
QMatrix qtranspose(QMatrix q);
Matrix dequantize(QMatrix q);
QMatrix qmat(const Buffer* weights, const Buffer* scales, size_t rows, size_t cols);

#define assume_shape(m, r, c) assert(((int)(r) == -1 || (m).rows == (size_t)(r)) && ((int)(c) == -1 || (m).cols == (size_t)(c)))

#endif // MATRIX_H

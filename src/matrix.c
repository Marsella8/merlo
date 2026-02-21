#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "utils.h"
#include <stddef.h>

float* at_f32(Matrix mat, size_t r, size_t c) {
#ifdef SAFETY
    if (r >= mat.rows || c >= mat.cols) {
        fprintf(stderr, "Error: Index out of bounds in at_f32\n");
        exit(1);
    }
#endif
    
    size_t start = mat.offset;
    start += r * mat.row_stride;
    start += c * mat.col_stride;
    return &((float*)mat.buffer->data)[start];
}

int8_t* at_i8(QMatrix mat, size_t r, size_t c) {
#ifdef SAFETY
    if (r >= mat.rows || c >= mat.cols) {
        fprintf(stderr, "Error: Index out of bounds in at_i8\n");
        exit(1);
    }
#endif

    size_t start = mat.offset;
    start += r * mat.row_stride;
    start += c * mat.col_stride;
    return &((int8_t*)mat.weights->data)[start];
}

static inline bool is_contiguous(Matrix m) {
    return m.row_stride >= 0 && m.col_stride == 1 && (size_t)m.row_stride == m.cols;
}

static inline float* as_ptr(Matrix mat) {
#ifdef SAFETY
    if (mat.buffer == NULL) {
        panic("as_ptr called on NULL matrix buffer");
    }
    assert(is_contiguous(mat));
#endif
    return &((float*)mat.buffer->data)[mat.offset];
}

Buffer* buf(size_t size) {
    Buffer* b_ptr = malloc(sizeof(Buffer));
#ifdef SAFETY
    assert(b_ptr != NULL);
#endif
    b_ptr->data = malloc(size);
#ifdef SAFETY
    assert(b_ptr->data != NULL);
#endif
    b_ptr->size = size;
    return b_ptr;
}

void free_buf(Buffer* b_ptr) {
    free(b_ptr->data);
    free(b_ptr);
}

void free_mat(Matrix m) {
#ifdef SAFETY
    if (m.buffer == NULL) {
        panic("freeing NULL buffer? bro ...");
    }
    if (!m.owned) {
        warning("freeing a non-owned matrix. remove this ...");
        return;
    }
#endif
    free_buf(m.buffer);
}

void free_qmat(QMatrix q) {
#ifdef SAFETY
    if (q.weights == NULL || q.scales == NULL) {
        panic("freeing NULL qmatrix buffers? bro ...");
    }
    if (!q.owned) {
        warning("freeing a non-owned qmatrix. remove this ...");
        return;
    }
#endif
    free_buf(q.weights);
    free_buf(q.scales);
}

Matrix mat(const Buffer* buffer, size_t rows, size_t cols) {
    Matrix mat = {
        .buffer = (Buffer*)buffer,
        .rows = rows,
        .cols = cols,
        .row_stride = cols,
        .col_stride = 1,
        .offset = 0,
        .owned = true
    };
    return mat;
}

Matrix empty(size_t rows, size_t cols) {
    Buffer* b = buf(rows * cols * sizeof(float));
    return mat(b, rows, cols);
}

Matrix mat_from_array(size_t rows, size_t cols, float m[rows][cols]) {
    size_t size = rows * cols * sizeof(float);
    Buffer* buffer = buf(size);
    memcpy(buffer->data, m, size);
    return mat(buffer, rows, cols);
}

bool eq(Matrix a, Matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        return 0;
    }
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            if (*at(a, i, j) != *at(b, i, j)) {
                return 0;
            }
        }
    }
    return 1;
}

void copy(Matrix src, Matrix dst) {
#ifdef SAFETY
    assume_shape(src, dst.rows, dst.cols);
#endif
    for (size_t i = 0; i < src.rows; i++) {
        for (size_t j = 0; j < src.cols; j++) {
            *at(dst, i, j) = *at(src, i, j);
        }
    }
}

Matrix transpose(Matrix mat) {
    Matrix transposed = {
        .buffer = mat.buffer,
        .rows = mat.cols,
        .cols = mat.rows,
        .row_stride = mat.col_stride,
        .col_stride = mat.row_stride,
        .offset = mat.offset,
        .owned = false
    };
    return transposed;
}

Matrix slice(Matrix mat, size_t r_start, size_t r_end, size_t c_start, size_t c_end) {
#ifdef SAFETY
    assert(r_start <= r_end && r_end <= mat.rows);
    assert(c_start <= c_end && c_end <= mat.cols);
#endif
    Matrix s = {
        .buffer = mat.buffer,
        .rows = r_end - r_start,
        .cols = c_end - c_start,
        .row_stride = mat.row_stride,
        .col_stride = mat.col_stride,
        .offset = mat.offset + r_start * mat.row_stride + c_start * mat.col_stride,
        .owned = false
    };
    return s;
}

Matrix add(Matrix a, Matrix b) {
#ifdef SAFETY
    assume_shape(a, b.rows, b.cols);
#endif
    Matrix result = empty(a.rows, a.cols);

    for (size_t r = 0; r < a.rows; r++) {
        for (size_t c = 0; c < a.cols; c++) {
            *at(result, r, c) = *at(a, r, c) + *at(b, r, c);
        }
    }
    
    return result;
}

Matrix scale(Matrix a, float value) {
    Matrix result = empty(a.rows, a.cols);
    for (size_t r = 0; r < a.rows; r++) {
        for (size_t c = 0; c < a.cols; c++) {
            *at(result, r, c) = *at(a, r, c) * value;
        }
    }
    return result;
}

Matrix matmul(Matrix a, Matrix b) {
#ifdef SAFETY
    assert(a.cols == b.rows);
#endif
    Matrix result = empty(a.rows, b.cols);

    if (is_contiguous(a) && is_contiguous(b)) {
        const float* a_data = as_ptr(a);
        const float* b_data = as_ptr(b);
        float* out_data = as_ptr(result);

        for (size_t i = 0; i < a.rows; i++) {
            for (size_t j = 0; j < b.cols; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < a.cols; k++) {
                    sum += a_data[i * a.cols + k] * b_data[k * b.cols + j];
                }
                out_data[i * b.cols + j] = sum;
            }
        }
        return result;
    }

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols; k++) {
                sum += *at(a, i, k) * *at(b, k, j);
            }
            *at(result, i, j) = sum;
        }
    }
    
    return result;
}

Matrix masked_matmul(Matrix a, Matrix b) {
 #ifdef SAFETY
    assert(a.cols == b.rows);
 #endif
    Matrix result = empty(a.rows, b.cols);
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols; k++) {
                sum += *at(a, i, k) * *at(b, k, j);
            }
            *at(result, i, j) = sum;
        }
        for (size_t j = i + 1; j < b.cols; j++) {
            *at(result, i, j) = -INFINITY;
        }
    }
    return result;
}

Matrix qmatmul(Matrix a, QMatrix b) {
    //TODO: make more performant by doing it directly
    Matrix dq = dequantize(b);
    Matrix m = matmul(a, dq);
    free_mat(dq);
    return m;
}

Matrix dequantize(QMatrix q) {
    Matrix out_m = empty(q.rows, q.cols);
    float* out = (float*)out_m.buffer->data;
    int8_t* weights = (int8_t*)q.weights->data;
    float* scales = (float*)q.scales->data;

    for (size_t i = 0; i < q.rows; i++) {
        for (size_t j = 0; j < q.cols; j++) {
            size_t src_idx = q.offset + i * q.row_stride + j * q.col_stride;
            
            size_t block_idx = src_idx / 32;
            float scale = scales[block_idx];
            
            size_t dst_idx = i * q.cols + j;
            out[dst_idx] = scale * weights[src_idx];
        }
    }
    return out_m;
}

QMatrix qmat(const Buffer* weights, const Buffer* scales, size_t rows, size_t cols) {
    QMatrix q = {
        .weights = (Buffer*)weights,
        .scales = (Buffer*)scales,
        .rows = rows,
        .cols = cols,
        .row_stride = cols,
        .col_stride = 1,
        .offset = 0,
        .owned = true
    };
    return q;
}

QMatrix qtranspose(QMatrix q) {
    QMatrix transposed = {
        .weights = q.weights,
        .scales = q.scales,
        .rows = q.cols,
        .cols = q.rows,
        .row_stride = q.col_stride,
        .col_stride = q.row_stride,
        .offset = q.offset,
        .owned = false
    };
    
    return transposed;
}

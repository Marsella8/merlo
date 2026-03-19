#include <math.h>
#include <string.h>

#include "kernels/fp32_matvec/kernel.h"
#include "kernels/gpu.h"
#include "kernels/q4_0_matvec/kernel.h"
#include "kernels/scale_matrix/kernel.h"
#include "matrix.h"
#include "model.h"
#include "utils.h"

float* at_f32(Matrix mat, size_t r, size_t c) {
#ifdef SAFETY
    if (r >= mat.rows || c >= mat.cols) {
        panic("%s:%d: Index out of bounds in at_f32\n", __FILE__, __LINE__);
    }
#endif
    
    size_t start = mat.offset;
    start += r * mat.row_stride;
    start += c * mat.col_stride;
    return &((float*)mat.buffer->data)[start];
}

static inline bool is_row_major(Matrix m) {
    return m.row_stride > 0 && m.col_stride == 1;
}

static inline bool is_col_major(Matrix m) {
    return m.col_stride > 0 && m.row_stride == 1;
}

static inline void assert_owned_shape(Matrix m, size_t rows, size_t cols) {
#ifdef SAFETY
    assume_shape(m, rows, cols);
    assert(m.buffer != NULL);
    assert(m.owned);
#else
    (void)m;
    (void)rows;
    (void)cols;
#endif
}

static float load_f32(const uint8_t *src) {
    float out;
    memcpy(&out, src, sizeof(out));
    return out;
}

static uint32_t load_u32(const uint8_t *src) {
    uint32_t out;
    memcpy(&out, src, sizeof(out));
    return out;
}


Buffer* buf(size_t size) {
    Buffer* b_ptr = malloc(sizeof(Buffer));
#ifdef SAFETY
    assert(b_ptr != NULL);
#endif
    size_t alloc_size = size == 0 ? 1 : size; // sometimes we construct empty matrices which kinda act as nullopt, but this trips up the allocator so we do this...
    b_ptr->data = malloc(alloc_size);
#ifdef SAFETY
    assert(b_ptr->data != NULL);
#endif
    b_ptr->size = size;
    b_ptr->owned = true;
    return b_ptr;
}

Buffer* watch_buf(void *data, size_t size) {
    Buffer *b_ptr = malloc(sizeof(Buffer));
#ifdef SAFETY
    assert(b_ptr != NULL);
#endif
    b_ptr->data = data;
    b_ptr->size = size;
    b_ptr->owned = false;
    return b_ptr;
}

Matrix stack_mat(Buffer *buffer, float *data, size_t rows, size_t cols) {
    *buffer = (Buffer){
        .data = data,
        .size = rows * cols * sizeof(float),
        .owned = false,
    };
    return (Matrix){
        .buffer = buffer,
        .rows = rows,
        .cols = cols,
        .row_stride = (int)cols,
        .col_stride = 1,
        .offset = 0,
        .owned = true,
    };
}

void free_buf(Buffer* b_ptr) {
    if (b_ptr->owned) {
        free(b_ptr->data);
    }
    free(b_ptr);
}

void free_mat(Matrix m) {
    if (m.buffer == NULL) {
#ifdef SAFETY
        panic("freeing NULL buffer? bro ...");
#else
        return;
#endif
    }
    if (!m.owned) {
#ifdef SAFETY
        warning("freeing a non-owned matrix. remove this ...");
#endif
        return;
    }
    free_buf(m.buffer);
}

void free_qmat(QMatrix q) {
    if (q.buffer == NULL) {
#ifdef SAFETY
        panic("freeing NULL qmatrix buffer? bro ...");
#else
        return;
#endif
    }
    free_buf(q.buffer);
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

size_t num_elements(Matrix m) {
    return m.rows * m.cols;
}

size_t num_bytes(Matrix m) {
    return num_elements(m) * sizeof(float);
}

size_t qnum_bytes_for_shape(size_t rows, size_t cols) {
    enum {
        Q4_ROWS_PER_PANEL = 16,
        Q4_COLS_PER_BLOCK = 32,
        Q4_BLOCK_BYTES = 320,
    };
    return ceil_div(rows, Q4_ROWS_PER_PANEL)
         * ceil_div(cols, Q4_COLS_PER_BLOCK)
         * Q4_BLOCK_BYTES;
}

Matrix empty(size_t rows, size_t cols) {
    Buffer* b = buf(rows * cols * sizeof(float));
    return mat(b, rows, cols);
}

Matrix as_contiguous(Matrix mat) {
    Buffer* b = buf(num_bytes(mat));
    float* dst = (float*)b->data;
    for (size_t r = 0; r < mat.rows; r++) {
        for (size_t c = 0; c < mat.cols; c++) {
            *dst++ = *at_f32(mat, r, c);
        }
    }
    return (Matrix){
        .buffer = b,
        .rows = mat.rows,
        .cols = mat.cols,
        .row_stride = (int)mat.cols,
        .col_stride = 1,
        .offset = 0,
        .owned = true,
    };
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
    add_into(a, b, result);
    return result;
}

void add_into(Matrix a, Matrix b, Matrix out) {
#ifdef SAFETY
    assume_shape(a, b.rows, b.cols);
    assert_owned_shape(out, a.rows, a.cols);
#endif
    for (size_t r = 0; r < a.rows; r++) {
        for (size_t c = 0; c < a.cols; c++) {
            *at(out, r, c) = *at(a, r, c) + *at(b, r, c);
        }
    }
}

Matrix scale(Matrix a, float value) {
    Matrix result = empty(a.rows, a.cols);
    scale_into(a, value, result);
    return result;
}

void scale_into(Matrix a, float value, Matrix out) {
#ifdef SAFETY
    assert_owned_shape(out, a.rows, a.cols);
#endif
    unsigned rows_pad = gpu_pad_qpu((unsigned)a.rows);
    size_t panel_elems = (size_t)rows_pad * (size_t)a.cols;
    if (panel_elems > SCALE_MATRIX_MAX_PANEL_FLOATS) {
        for (size_t r = 0; r < a.rows; r++) {
            for (size_t c = 0; c < a.cols; c++)
                *at(out, r, c) = *at(a, r, c) * value;
        }
        return;
    }
    scale_matrix_into(a, value, out);
}

void scale_inplace(Matrix a, float value) {
    scale_into(a, value, a);
}

void matmul_into(Matrix a, Matrix b, Matrix out) {
#ifdef SAFETY
    assert(a.cols == b.rows);
    assert_owned_shape(out, a.rows, b.cols);
#endif
    if (a.cols > INTERMEDIATE_SIZE) {
        for (size_t i = 0; i < a.rows; i++) {
            for (size_t j = 0; j < b.cols; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < a.cols; k++) {
                    sum += *at(a, i, k) * *at(b, k, j);
                }
                *at(out, i, j) = sum;
            }
        }
        return;
    }
    fp32_matmul_into(a, b, out);
}

Matrix matmul(Matrix a, Matrix b) {
    Matrix result = empty(a.rows, b.cols);
    matmul_into(a, b, result);
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
    Matrix out = empty(a.rows, b.rows);
    qmatmul_into(a, b, out);
    return out;
}

static float q4_0_decode_scalar(const QMatrix *q, size_t r, size_t c) {
    enum {
        Q4_ROWS_PER_PANEL = 16,
        Q4_COLS_PER_BLOCK = 32,
        Q4_SECTION_BYTES = 64,
        Q4_BLOCK_BYTES = 320,
        Q4_NIBBLES_PER_WORD = 8,
    };
#ifdef SAFETY
    assert(q->buffer != NULL);
    assert(q->buffer->data != NULL);
    assert(r < q->rows);
    assert(c < q->cols);
#endif
    size_t blocks_per_panel = ceil_div(q->cols, Q4_COLS_PER_BLOCK);
    size_t total_bytes = qnum_bytes_for_shape(q->rows, q->cols);
#ifdef SAFETY
    assert(q->buffer->size >= total_bytes);
#endif

    size_t panel = r / Q4_ROWS_PER_PANEL;
    size_t lane = r % Q4_ROWS_PER_PANEL;
    size_t block = c / Q4_COLS_PER_BLOCK;
    size_t nibble = c % Q4_COLS_PER_BLOCK;
    size_t word = nibble / Q4_NIBBLES_PER_WORD;
    size_t nibble_shift = 4 * (nibble % Q4_NIBBLES_PER_WORD);
    size_t panel_stride = blocks_per_panel * Q4_BLOCK_BYTES;
    size_t block_base = panel * panel_stride + block * Q4_BLOCK_BYTES;
    const uint8_t *base = q->buffer->data;

    float scale = load_f32(base + block_base + lane * sizeof(float));
    uint32_t packed = load_u32(base + block_base
                                     + Q4_SECTION_BYTES * (1 + word)
                                     + lane * sizeof(uint32_t));
    uint32_t qv = (packed >> nibble_shift) & 0xFu;
    return ((float)((int)qv - 8)) * scale;
}

void qmatrix_row_decode_into(QMatrix q, size_t row, Matrix dst) {
#ifdef SAFETY
    assert(row < q.rows);
    assume_shape(dst, 1, q.cols);
#endif
    for (size_t c = 0; c < q.cols; c++)
        *at(dst, 0, c) = q4_0_decode_scalar(&q, row, c);
}

static float q4_matmul_row_x[INTERMEDIATE_SIZE] __attribute__((aligned(16)));

void qmatmul_into(Matrix a, QMatrix b, Matrix out) {
#ifdef SAFETY
    assert(a.cols == b.cols);
    assert_owned_shape(out, a.rows, b.rows);
    assert(a.cols <= INTERMEDIATE_SIZE);
#endif
    Buffer xb;
    for (size_t i = 0; i < a.rows; i++) {
        if (a.row_stride > 0 && a.col_stride == 1) {
            const float *src =
                &((float *)a.buffer->data)[a.offset + i * (size_t)a.row_stride];
            memcpy(q4_matmul_row_x, src, a.cols * sizeof(float));
        } else {
            for (size_t k = 0; k < a.cols; k++)
                q4_matmul_row_x[k] = *at_f32(a, i, k);
        }
        Matrix x = stack_mat(&xb, q4_matmul_row_x, 1, a.cols);
        Matrix o_row = slice(out, i, i + 1, 0, b.rows);
        q4_0_matvec_into(x, b, o_row);
    }
}

void matvec_into(Matrix x, Matrix w, Matrix out) {
#ifdef SAFETY
    assume_shape(x, 1, w.cols);
    assume_shape(out, 1, w.rows);
#endif
    if (w.cols > INTERMEDIATE_SIZE) {
        for (size_t j = 0; j < w.rows; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < w.cols; k++)
                sum += *at_f32(x, 0, k) * *at_f32(w, j, k);
            *at_f32(out, 0, j) = sum;
        }
        return;
    }
    fp32_matvec_into(x, w, out);
}

void qmatvec_into(Matrix x, QMatrix b, Matrix out) {
#ifdef SAFETY
    assume_shape(x, 1, b.cols);
    assume_shape(out, 1, b.rows);
    assert(b.cols <= INTERMEDIATE_SIZE);
#endif
    q4_0_matvec_into(x, b, out);
}

Matrix dequantize(QMatrix q) {
    Matrix out = empty(q.rows, q.cols);
    for (size_t r = 0; r < q.rows; r++) {
        for (size_t c = 0; c < q.cols; c++) {
            *at(out, r, c) = q4_0_decode_scalar(&q, r, c);
        }
    }
    return out;
}

QMatrix qmat(const Buffer* buffer, size_t rows, size_t cols) {
    return (QMatrix){
        .buffer = (Buffer*)buffer,
        .rows = rows,
        .cols = cols,
    };
}

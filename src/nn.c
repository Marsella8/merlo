#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "nn.h"
#include "model.h"
#include "utils.h"
#include <string.h>
#include "matrix.h"

Matrix silu(Matrix x) {
    Matrix m = empty(x.rows, x.cols);
    
    for (size_t i = 0; i < x.rows; i++) {
        for (size_t j = 0; j < x.cols; j++) {
            float val = *at(x, i, j);
            float sigmoid = 1.0f / (1.0f + expf(-val));
            *at(m, i, j) = val * sigmoid;
        }
    }
    
    return m;
}


Matrix embed(Matrix embeddings, Matrix tokens) {
#ifdef SAFETY
    assert(tokens.rows == 1);
#endif
    Matrix m = empty(tokens.cols, embeddings.cols);
    for (size_t i = 0; i < tokens.cols; i++) {
        size_t id = (size_t)*at(tokens, 0, i);
        memcpy(at(m, i, 0), at(embeddings, id, 0), embeddings.cols * sizeof(float));
    }
    return m;
}

Matrix rope(Matrix x, size_t pos) {
    Matrix out = empty(x.rows, x.cols);
    for (size_t t = 0; t < x.rows; t++) {
        for (size_t i = 0; i < x.cols; i += 2) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(i % HEAD_DIM) / HEAD_DIM);
            float angle = (pos + t) * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float x0 = *at(x, t, i);
            float x1 = *at(x, t, i + 1);
            *at(out, t, i)     = x0 * cos_a - x1 * sin_a;
            *at(out, t, i + 1) = x0 * sin_a + x1 * cos_a;
        }
    }
    return out;
}

Matrix rms_norm(Matrix x, Matrix weight) {
    Matrix m = empty(x.rows, x.cols);
    for (size_t i=0; i<x.rows; i++) {
        float rms = 0;
        for (size_t j=0; j<x.cols; j++) {
            rms+=*at(x,i,j) * *at(x,i,j);
        }
        rms = rms/x.cols;
        rms += RMS_NORM_EPS;
        float i_rms = 1.0f / sqrtf(rms); // sub with the cool quake III algo hahaha
        for (size_t j=0; j<x.cols; j++) {
            *at(m, i, j) = *at(x, i, j) * *at(weight, 0, j) * i_rms;
        }
    }
    return m;
}

Matrix decode_gqa(Matrix x, Matrix Wq, Matrix Wk, Matrix Wv, Matrix Wo, LayerCache cache, size_t pos) {
    size_t T = pos + 1;
#ifdef SAFETY
    assume_shape(x, 1, HIDDEN_SIZE);
    assume_shape(Wq, HIDDEN_SIZE, HIDDEN_SIZE);
    assume_shape(Wk, HIDDEN_SIZE, KV_SIZE);
    assume_shape(Wv, HIDDEN_SIZE, KV_SIZE);
    assume_shape(Wo, HIDDEN_SIZE, HIDDEN_SIZE);
#endif

    Matrix q_proj = matmul(x, Wq);
    Matrix k_proj = matmul(x, Wk);
    Matrix v = matmul(x, Wv);

    Matrix q = rope(q_proj, pos);
    Matrix k = rope(k_proj, pos);
    free_mat(q_proj);
    free_mat(k_proj);

#ifdef SAFETY
    assume_shape(q, 1, HIDDEN_SIZE);
    assume_shape(k, 1, KV_SIZE);
    assume_shape(v, 1, KV_SIZE);
    assume_shape(cache.k, KV_SIZE, -1);
    assume_shape(cache.v, KV_SIZE, -1);
    assert(T <= cache.k.cols);
    assert(T <= cache.v.cols);
#endif

    copy(transpose(k), slice(cache.k, 0, KV_SIZE, pos, pos + 1));
    copy(transpose(v), slice(cache.v, 0, KV_SIZE, pos, pos + 1));

    Matrix O = empty(1, HIDDEN_SIZE);
    for (size_t qh = 0; qh < NUM_Q_HEADS; qh++) {
        size_t kvh = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        Matrix Qh = slice(q, 0, 1, qh * HEAD_DIM, (qh + 1) * HEAD_DIM);
        Matrix Kh = slice(cache.k, kvh * HEAD_DIM, (kvh + 1) * HEAD_DIM, 0, T);
        Matrix Vh = slice(cache.v, kvh * HEAD_DIM, (kvh + 1) * HEAD_DIM, 0, T);
        Matrix scores = matmul(Qh, Kh); // 1 x T
        Matrix attn = softmax(scores);
        Matrix Hh = matmul(attn, transpose(Vh)); // 1 x HEAD_DIM
        Matrix out_h = slice(O, 0, 1, qh * HEAD_DIM, (qh + 1) * HEAD_DIM);
        copy(Hh, out_h);
        free_mat(scores);
        free_mat(attn);
        free_mat(Hh);
    }

    free_mat(q);
    free_mat(k);
    free_mat(v);

    Matrix out = matmul(O, Wo);
    free_mat(O);
    return out;
}

Matrix prefill_gqa(Matrix x, Matrix Wq, Matrix Wk, Matrix Wv, Matrix Wo, LayerCache cache, size_t pos) {
    size_t T = x.rows;
#ifdef SAFETY
    assume_shape(x, T, HIDDEN_SIZE);
    assume_shape(Wq, HIDDEN_SIZE, HIDDEN_SIZE);
    assume_shape(Wk, HIDDEN_SIZE, KV_SIZE);
    assume_shape(Wv, HIDDEN_SIZE, KV_SIZE);
    assume_shape(Wo, HIDDEN_SIZE, HIDDEN_SIZE);
#endif

    Matrix Q_proj = matmul(x, Wq);
    Matrix K_proj = matmul(x, Wk);
    Matrix V = matmul(x, Wv);

    Matrix Q = rope(Q_proj, pos);
    Matrix K = rope(K_proj, pos);
    free_mat(Q_proj);
    free_mat(K_proj);

#ifdef SAFETY
    assume_shape(Q, T, HIDDEN_SIZE);
    assume_shape(K, T, KV_SIZE);
    assume_shape(V, T, KV_SIZE);
    assume_shape(cache.k, KV_SIZE, -1);
    assume_shape(cache.v, KV_SIZE, -1);
    assert(pos + T <= cache.k.cols);
    assert(pos + T <= cache.v.cols);
#endif

    copy(transpose(K), slice(cache.k, 0, KV_SIZE, pos, pos + T));
    copy(transpose(V), slice(cache.v, 0, KV_SIZE, pos, pos + T));

    Matrix O = empty(T, HIDDEN_SIZE);
    for (size_t qh = 0; qh < NUM_Q_HEADS; qh++) {
        size_t kvh = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        Matrix Qh = slice(Q, 0, T, qh * HEAD_DIM, (qh + 1) * HEAD_DIM);
        Matrix Kh = slice(K, 0, T, kvh * HEAD_DIM, (kvh + 1) * HEAD_DIM);
        Matrix Vh = slice(V, 0, T, kvh * HEAD_DIM, (kvh + 1) * HEAD_DIM);
        Matrix scores = masked_matmul(Qh, transpose(Kh)); // T x T (causal)
        Matrix attn = softmax(scores);
        Matrix Hh = matmul(attn, Vh); // T x HEAD_DIM
        Matrix out_h = slice(O, 0, T, qh * HEAD_DIM, (qh + 1) * HEAD_DIM);
        copy(Hh, out_h);
        free_mat(scores);
        free_mat(attn);
        free_mat(Hh);
    }

    free_mat(Q);
    free_mat(K);
    free_mat(V);

    Matrix out = matmul(O, Wo);
    free_mat(O);
    return out;
}

Matrix elementwise(Matrix a, Matrix b) {
#ifdef SAFETY
    assume_shape(a, b.rows, b.cols);
#endif
    Matrix m = empty(a.rows, a.cols);
    
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            *at(m, i, j) = *at(a, i, j) * *at(b, i, j);
        }
    }
    
    return m;
}

Matrix ffn(Matrix x, Matrix gate, Matrix up, Matrix down) {
    Matrix m = matmul(x, gate);
    Matrix xgate = silu(m);
    free_mat(m);

    Matrix xup = matmul(x, up);
    Matrix ew = elementwise(xgate, xup);
    free_mat(xgate);
    free_mat(xup);
    
    Matrix out = matmul(ew, down);
    free_mat(ew);
    
    return out;
}

Matrix softmax(Matrix x) {
    Matrix out = empty(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; i++) {
        float max_val = -INFINITY;
        for (size_t j = 0; j < x.cols; j++) {
            if (*at(x, i, j) > max_val) {
                max_val = *at(x, i, j);
            }
        }
        float sum = 0.0f;
        for (size_t j = 0; j < x.cols; j++) {
            sum += expf(*at(x, i, j) - max_val);
        }
        for (size_t j = 0; j < x.cols; j++) {
            *at(out, i, j) = expf(*at(x, i, j) - max_val) / sum;
        }
    }
    return out;
}

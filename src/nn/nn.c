#include <math.h>

#include "kernels/gating/kernel.h"
#include "kernels/silu/kernel.h"
#include "kernels/softmax/kernel.h"
#include "nn.h"

static inline void assert_owned_shape(Matrix m, size_t rows, size_t cols) {
#ifdef SAFETY
    assume_shape(m, rows, cols);
    assert(m.buffer != NULL);
    assert(m.owned);
#endif
}

static float rope_freq[HEAD_DIM / 2];
static bool rope_freq_ready = false;

static void rope_into(Matrix x, size_t pos, Matrix out) {
#ifdef SAFETY
    assume_shape(out, x.rows, x.cols);
    assert(x.cols % HEAD_DIM == 0);
#endif
    if (!rope_freq_ready) {
        for (size_t i = 0; i < HEAD_DIM / 2; i++)
            rope_freq[i] = 1.0f / powf(ROPE_THETA, (2.0f * (float)i) / HEAD_DIM);
        rope_freq_ready = true;
    }
    for (size_t t = 0; t < x.rows; t++) {
        float token_pos = (float)(pos + t);
        for (size_t head = 0; head < x.cols; head += HEAD_DIM) {
            size_t half = HEAD_DIM / 2;
            for (size_t i = 0; i < half; i++) {
                float angle = token_pos * rope_freq[i];
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float x0 = *at(x, t, head + i);
                float x1 = *at(x, t, head + half + i);
                *at(out, t, head + i) = x0 * cos_a - x1 * sin_a;
                *at(out, t, head + half + i) = x1 * cos_a + x0 * sin_a;
            }
        }
    }
}

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

void silu_inplace(Matrix x) {
    silu_kernel_inplace(x);
}

void embed_into(QMatrix lm_head, Matrix tokens, Matrix out) {
#ifdef SAFETY
    assert(tokens.rows == 1);
    assume_shape(lm_head, VOCAB_SIZE, HIDDEN_SIZE);
    assume_shape(out, tokens.cols, HIDDEN_SIZE);
#endif
    for (size_t i = 0; i < tokens.cols; i++) {
        size_t id = (size_t)*at(tokens, 0, i);
#ifdef SAFETY
        assert(id < lm_head.rows);
#endif
        Matrix out_row = slice(out, i, i + 1, 0, HIDDEN_SIZE);
        qmatrix_row_decode_into(lm_head, id, out_row);
    }
}

Matrix rope(Matrix x, size_t pos) {
    Matrix out = empty(x.rows, x.cols);
    rope_into(x, pos, out);
    return out;
}

void rms_norm_into(Matrix x, Matrix weight, Matrix out) {
#ifdef SAFETY
    assume_shape(weight, 1, x.cols);
    assert_owned_shape(out, x.rows, x.cols);
#endif
    for (size_t i=0; i<x.rows; i++) {
        float rms = 0;
        for (size_t j=0; j<x.cols; j++) {
            rms+=*at(x,i,j) * *at(x,i,j);
        }
        rms = rms/x.cols;
        rms += RMS_NORM_EPS;
        float i_rms = 1.0f / sqrtf(rms);
        for (size_t j=0; j<x.cols; j++) {
            *at(out, i, j) = *at(x, i, j) * *at(weight, 0, j) * i_rms;
        }
    }
}

void decode_gqa_into(Matrix x,
                     QMatrix Wq,
                     QMatrix Wk,
                     QMatrix Wv,
                     QMatrix Wo,
                     LayerCache cache,
                     size_t pos,
                     Matrix out) {
    size_t T = pos + 1;
#ifdef SAFETY
    assume_shape(x, 1, HIDDEN_SIZE);
    assume_shape(Wq, HIDDEN_SIZE, HIDDEN_SIZE);
    assume_shape(Wk, KV_SIZE, HIDDEN_SIZE);
    assume_shape(Wv, KV_SIZE, HIDDEN_SIZE);
    assume_shape(Wo, HIDDEN_SIZE, HIDDEN_SIZE);
    assert(T <= MAX_SEQ_LEN);
    assert_owned_shape(out, 1, HIDDEN_SIZE);
#endif

    float q_proj_data[HIDDEN_SIZE];
    float k_proj_data[KV_SIZE];
    float v_data[KV_SIZE];
    float q_data[HIDDEN_SIZE];
    float k_data[KV_SIZE];
    float o_data[HIDDEN_SIZE];
    float scores_data[MAX_SEQ_LEN];
    Buffer q_proj_buf;
    Buffer k_proj_buf;
    Buffer v_buf;
    Buffer q_buf;
    Buffer k_buf;
    Buffer o_buf;
    Matrix q_proj = stack_mat(&q_proj_buf, q_proj_data, 1, HIDDEN_SIZE);
    Matrix k_proj = stack_mat(&k_proj_buf, k_proj_data, 1, KV_SIZE);
    Matrix v = stack_mat(&v_buf, v_data, 1, KV_SIZE);
    Matrix q = stack_mat(&q_buf, q_data, 1, HIDDEN_SIZE);
    Matrix k = stack_mat(&k_buf, k_data, 1, KV_SIZE);
    Matrix O = stack_mat(&o_buf, o_data, 1, HIDDEN_SIZE);

    qmatvec_into(x, Wq, q_proj);
    qmatvec_into(x, Wk, k_proj);
    qmatvec_into(x, Wv, v);
    rope_into(q_proj, pos, q);
    rope_into(k_proj, pos, k);

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

    float *ck_base = (float *)cache.k.buffer->data + cache.k.offset;
    float *cv_base = (float *)cache.v.buffer->data + cache.v.offset;
    size_t cache_stride = (size_t)cache.k.row_stride;
    const float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (size_t qh = 0; qh < NUM_Q_HEADS; qh++) {
        size_t kvh = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        float *qh_ptr = q_data + qh * HEAD_DIM;
        float *oh_ptr = o_data + qh * HEAD_DIM;
        size_t kv_off = kvh * HEAD_DIM;

        for (size_t t = 0; t < T; t++)
            scores_data[t] = 0.0f;
        for (size_t i = 0; i < HEAD_DIM; i++) {
            float qi = qh_ptr[i] * attn_scale;
            float *k_row = ck_base + (kv_off + i) * cache_stride;
            for (size_t t = 0; t < T; t++)
                scores_data[t] += qi * k_row[t];
        }

        float max_s = scores_data[0];
        for (size_t t = 1; t < T; t++)
            if (scores_data[t] > max_s) max_s = scores_data[t];
        float sum = 0.0f;
        for (size_t t = 0; t < T; t++) {
            scores_data[t] = expf(scores_data[t] - max_s);
            sum += scores_data[t];
        }
        float inv_sum = 1.0f / sum;
        for (size_t t = 0; t < T; t++)
            scores_data[t] *= inv_sum;

        for (size_t i = 0; i < HEAD_DIM; i++) {
            float *v_row = cv_base + (kv_off + i) * cache_stride;
            float val = 0.0f;
            for (size_t t = 0; t < T; t++)
                val += scores_data[t] * v_row[t];
            oh_ptr[i] = val;
        }
    }
    qmatvec_into(O, Wo, out);
}

Matrix prefill_gqa(Matrix x, QMatrix Wq, QMatrix Wk, QMatrix Wv, QMatrix Wo, LayerCache cache) {
    size_t T = x.rows;
#ifdef SAFETY
    assume_shape(x, T, HIDDEN_SIZE);
    assume_shape(Wq, HIDDEN_SIZE, HIDDEN_SIZE);
    assume_shape(Wk, KV_SIZE, HIDDEN_SIZE);
    assume_shape(Wv, KV_SIZE, HIDDEN_SIZE);
    assume_shape(Wo, HIDDEN_SIZE, HIDDEN_SIZE);
#endif

    Matrix Q_proj = empty(T, HIDDEN_SIZE);
    Matrix K_proj = empty(T, KV_SIZE);
    Matrix V = empty(T, KV_SIZE);
    qmatmul_into(x, Wq, Q_proj);
    qmatmul_into(x, Wk, K_proj);
    qmatmul_into(x, Wv, V);

    Matrix Q = rope(Q_proj, 0);
    Matrix K = rope(K_proj, 0);
    free_mat(Q_proj);
    free_mat(K_proj);

#ifdef SAFETY
    assume_shape(Q, T, HIDDEN_SIZE);
    assume_shape(K, T, KV_SIZE);
    assume_shape(V, T, KV_SIZE);
    assume_shape(cache.k, KV_SIZE, -1);
    assume_shape(cache.v, KV_SIZE, -1);
    assert(T <= cache.k.cols);
    assert(T <= cache.v.cols);
#endif

    copy(transpose(K), slice(cache.k, 0, KV_SIZE, 0, T));
    copy(transpose(V), slice(cache.v, 0, KV_SIZE, 0, T));

    Matrix O = empty(T, HIDDEN_SIZE);
    const float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (size_t qh = 0; qh < NUM_Q_HEADS; qh++) {
        size_t kvh = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        Matrix Qh = slice(Q, 0, T, qh * HEAD_DIM, (qh + 1) * HEAD_DIM);
        Matrix Kh = slice(cache.k, kvh * HEAD_DIM, (kvh + 1) * HEAD_DIM, 0, T);
        Matrix Vh = slice(cache.v, kvh * HEAD_DIM, (kvh + 1) * HEAD_DIM, 0, T);
        Matrix raw_scores = masked_matmul(Qh, Kh); // T x T (causal)
        Matrix scores = scale(raw_scores, attn_scale);
        Matrix attn = softmax(scores);
        free_mat(raw_scores);
        Matrix Hh = matmul(attn, transpose(Vh)); // T x HEAD_DIM
        Matrix out_h = slice(O, 0, T, qh * HEAD_DIM, (qh + 1) * HEAD_DIM);
        copy(Hh, out_h);
        free_mat(scores);
        free_mat(attn);
        free_mat(Hh);
    }

    free_mat(Q);
    free_mat(K);
    free_mat(V);

    Matrix out = empty(T, HIDDEN_SIZE);
    qmatmul_into(O, Wo, out);
    free_mat(O);
    return out;
}

Matrix elementwise(Matrix a, Matrix b) {
#ifdef SAFETY
    assume_shape(a, b.rows, b.cols);
#endif
    Matrix m = empty(a.rows, a.cols);
    elementwise_mul_into(a, b, m);
    return m;
}

void elementwise_mul_into(Matrix a, Matrix b, Matrix out) {
#ifdef SAFETY
    assume_shape(a, b.rows, b.cols);
    assert_owned_shape(out, a.rows, a.cols);
#endif
    gating_kernel_into(a, b, out);
}

void ffn_decode_into(Matrix x, QMatrix gate, QMatrix up, QMatrix down, Matrix out) {
#ifdef SAFETY
    assume_shape(x, 1, HIDDEN_SIZE);
    assume_shape(gate, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    assume_shape(up, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    assume_shape(down, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    assert_owned_shape(out, 1, HIDDEN_SIZE);
#endif
    float m_data[INTERMEDIATE_SIZE];
    float xup_data[INTERMEDIATE_SIZE];
    Buffer m_buf;
    Buffer xup_buf;
    Matrix m = stack_mat(&m_buf, m_data, 1, INTERMEDIATE_SIZE);
    Matrix xup = stack_mat(&xup_buf, xup_data, 1, INTERMEDIATE_SIZE);

    qmatvec_into(x, gate, m);
    silu_kernel_inplace(m);
    qmatvec_into(x, up, xup);
    for (size_t i = 0; i < INTERMEDIATE_SIZE; i++)
        m_data[i] *= xup_data[i];
    qmatvec_into(m, down, out);
}

void ffn_prefill_into(Matrix x, QMatrix gate, QMatrix up, QMatrix down, Matrix out) {
#ifdef SAFETY
    assume_shape(gate, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    assume_shape(up, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    assume_shape(down, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    assert_owned_shape(out, x.rows, HIDDEN_SIZE);
#endif
    Matrix m = empty(x.rows, INTERMEDIATE_SIZE);
    Matrix xup = empty(x.rows, INTERMEDIATE_SIZE);
    Matrix ew = empty(x.rows, INTERMEDIATE_SIZE);

    qmatmul_into(x, gate, m);
    silu_inplace(m);
    qmatmul_into(x, up, xup);
    elementwise_mul_into(m, xup, ew);
    qmatmul_into(ew, down, out);
    free_mat(m);
    free_mat(xup);
    free_mat(ew);
}

Matrix softmax(Matrix x) {
    Matrix out = empty(x.rows, x.cols);
    softmax_into(x, out);
    return out;
}

void softmax_into(Matrix x, Matrix out) {
#ifdef SAFETY
    assert_owned_shape(out, x.rows, x.cols);
#endif
    softmax_kernel_into(x, out);
}

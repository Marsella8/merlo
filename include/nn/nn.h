#ifndef NN_H
#define NN_H

#include "matrix.h"
#include "model.h"

void embed_into(QMatrix lm_head, Matrix tokens, Matrix out);

Matrix rope(Matrix x, size_t pos);

void rms_norm_into(Matrix x, Matrix weight, Matrix out);

void ffn_prefill_into(Matrix x, QMatrix gate, QMatrix up, QMatrix down, Matrix out);
void ffn_decode_into(Matrix x, QMatrix gate, QMatrix up, QMatrix down, Matrix out);

Matrix silu(Matrix x);
void silu_inplace(Matrix x);

Matrix elementwise(Matrix a, Matrix b);
void elementwise_mul_into(Matrix a, Matrix b, Matrix out);

Matrix softmax(Matrix x);
void softmax_into(Matrix x, Matrix out);

void decode_gqa_into(Matrix x,
                     QMatrix Wq,
                     QMatrix Wk,
                     QMatrix Wv,
                     QMatrix Wo,
                     LayerCache cache,
                     size_t pos,
                     Matrix out);

Matrix prefill_gqa(Matrix x, QMatrix Wq, QMatrix Wk, QMatrix Wv, QMatrix Wo, LayerCache cache);

#endif // NN_H

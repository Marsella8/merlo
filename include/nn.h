#ifndef NN_H
#define NN_H

#include <stddef.h>
#include "matrix.h"
#include "model.h"

Matrix embed(Matrix embeddings, Matrix tokens);

Matrix rope(Matrix x, size_t pos);

Matrix rms_norm(Matrix x, Matrix weight);

Matrix ffn(Matrix x, Matrix gate, Matrix up, Matrix down);

Matrix silu(Matrix x);

Matrix elementwise(Matrix a, Matrix b);

Matrix softmax(Matrix x);

Matrix decode_gqa(Matrix x, Matrix Wq, Matrix Wk, Matrix Wv, Matrix Wo, LayerCache cache, size_t pos);

Matrix prefill_gqa(Matrix x, Matrix Wq, Matrix Wk, Matrix Wv, Matrix Wo, LayerCache cache, size_t pos);

#endif // NN_H

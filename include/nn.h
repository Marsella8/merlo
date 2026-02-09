#ifndef NN_H
#define NN_H

#include <stddef.h>
#include "matrix.h"
#include "model.h"

Matrix embed(Matrix embeddings, Matrix tokens);

Matrix rope(Matrix x, int pos);

Matrix rms_norm(Matrix x, Matrix weight);

Matrix ffn(Matrix x, Matrix gate, Matrix up, Matrix down);

Matrix silu(Matrix x);

Matrix elementwise(Matrix a, Matrix b);

Matrix softmax(Matrix x);

Matrix engine_gqa(Matrix x, Matrix q, Matrix k, Matrix v, Matrix o, LayerCache* cache, int pos);

Matrix prefill_gqa(Matrix x, Matrix q, Matrix k, Matrix v, Matrix o, int pos, LayerCache* cache);

#endif // NN_H


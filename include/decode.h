#ifndef DECODE_H
#define DECODE_H

#include "model.h"
#include "matrix.h"

Matrix decode_layer_fwd(Block b, Matrix x, LayerCache cache);
Matrix fwd(SmolLM2* model, size_t token_id);

#endif // DECODE_H

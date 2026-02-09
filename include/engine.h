#ifndef ENGINE_H
#define ENGINE_H

#include "model.h"
#include "matrix.h"

Matrix engine_layer_fwd(Block b, Matrix x, int pos, LayerCache* cache);
size_t fwd(SmolLM2 model, KVCache* cache, size_t token_id, size_t pos);

#endif // ENGINE_H


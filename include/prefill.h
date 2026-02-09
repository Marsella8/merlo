#ifndef PREFILL_H
#define PREFILL_H

#include "model.h"
#include "matrix.h"

Matrix prefill_layer_fwd(Block b, Matrix x, int pos, LayerCache* cache);
KVCache prefill(SmolLM2 model, Matrix x);

#endif // PREFILL_H


#ifndef LOAD_H
#define LOAD_H

#include <stddef.h>
#include <stdint.h>

#include "model.h"

Block load_block_at(uint8_t *layer_ptr);
Endpoint load_endpoint_at(uint8_t *embed_ptr, uint8_t *out_norm_ptr);
SmolLMLayerShard load_layer_shard_at(uint8_t *layer_ptr, size_t num_layers);

#endif

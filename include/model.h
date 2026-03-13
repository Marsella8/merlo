#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"

#define NUM_MAIN_LAYERS 30
#define NUM_SPEC_LAYERS 10 // TODO: idk this number yet 
#define VOCAB_SIZE 49152
#define HIDDEN_SIZE 576
#define INTERMEDIATE_SIZE 1536
#define HEAD_DIM 64
#define NUM_Q_HEADS 9
#define NUM_KV_HEADS 3
#define KV_SIZE (NUM_KV_HEADS * HEAD_DIM)
#define MAX_SEQ_LEN 2048
#define RMS_NORM_EPS 1e-05f
#define ROPE_THETA 100000.0f

typedef struct {
    Matrix k;
    Matrix v;
} LayerCache;

typedef struct {
    Matrix attn_norm;
    Matrix q, k, v, o;
    
    Matrix ffn_norm;
    Matrix gate, up, down;
    LayerCache cache;
} Block;

typedef struct {
    Matrix embeddings;
} SmolLMHeadShard;

typedef struct {
    size_t num_layers;
    Block* blocks;
} SmolLMLayerShard;

typedef struct {
    Matrix final_norm;
    Matrix lm_head;
} SmolLMTailShard;

typedef struct {
    SmolLMHeadShard head;
    SmolLMLayerShard layers;
    SmolLMTailShard tail;
} SmolLM2;

SmolLMHeadShard load_head_shard();
SmolLMLayerShard load_layer_shard(size_t num_layers);
SmolLMTailShard load_tail_shard();
void free_head_shard(SmolLMHeadShard head);
void free_layer_shard(SmolLMLayerShard layers);
void free_tail_shard(SmolLMTailShard tail);

Matrix head_fwd(SmolLMHeadShard head, Matrix token_ids);
Matrix layers_prefill_fwd(SmolLMLayerShard layers, Matrix x);
Matrix layers_decode_fwd(SmolLMLayerShard layers, Matrix x, size_t pos);
Matrix tail_fwd(SmolLMTailShard tail, Matrix x);

SmolLM2 load_model(size_t num_layers);
SmolLM2 load_main_model();
SmolLM2 load_spec_model();
void free_model(SmolLM2 model);
Matrix load_matrix(const char* name, const int layer);
QMatrix load_qmatrix(const char *name, const int layer);
void prefill(SmolLM2 model, Matrix x);
Matrix fwd(SmolLM2 model, size_t token_id, size_t pos);

#endif // MODEL_H

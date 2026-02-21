#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include "matrix.h"

#define NUM_LAYERS 30
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
    Matrix attn_norm;
    Matrix q, k, v, o;
    
    Matrix ffn_norm;
    Matrix gate, up, down;
} Block;

typedef struct {
    Matrix k;
    Matrix v;
} LayerCache;

typedef struct {
    LayerCache caches[NUM_LAYERS]; 
} KVCache;

typedef struct {
    Matrix embeddings;
    Block blocks[NUM_LAYERS];
    Matrix final_norm;
    Matrix lm_head;
    KVCache cache;
} SmolLM2;

SmolLM2 load_model();
void free_kvcache(KVCache cache);
void free_model(SmolLM2 model);
Matrix load_matrix(const char* name, const int layer);
QMatrix load_qmatrix(const char *name, const int layer);
void prefill(SmolLM2 model, Matrix x);
Matrix fwd(SmolLM2 model, size_t token_id, size_t pos);

#endif // MODEL_H

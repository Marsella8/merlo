#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"

#define NUM_LAYERS 30
#define MAX_NUM_LAYERS 30
#define VOCAB_SIZE 49152
#define HIDDEN_SIZE 576
#define INTERMEDIATE_SIZE 1536
#define HEAD_DIM 64
#define NUM_Q_HEADS 9
#define NUM_KV_HEADS 3
#define KV_SIZE (NUM_KV_HEADS * HEAD_DIM)
#define MAX_SEQ_LEN 256
#define RMS_NORM_EPS 1e-05f
#define ROPE_THETA 100000.0f

typedef struct {
    Matrix k;
    Matrix v;
} LayerCache;

typedef struct {
    Matrix attn_norm;
    QMatrix q;
    QMatrix k;
    QMatrix v;
    QMatrix o;
    Matrix ffn_norm;
    QMatrix gate;
    QMatrix up;
    QMatrix down;
    LayerCache cache;
} Block;

typedef struct {
    Matrix out_norm;
    QMatrix lm_head;
} Endpoint;

typedef struct {
    size_t num_layers;
    Block blocks[MAX_NUM_LAYERS];
} SmolLMLayerShard;

typedef struct {
    Endpoint endpoint;
    SmolLMLayerShard layers;
} Model;

Matrix block_prefill_fwd(Block b, Matrix x);
void block_decode_fwd_into(Block b, Matrix x, size_t pos, Matrix out);

/** Sum of per-layer attn vs FFN cycles across decode_profile_reset … next reset. */
void decode_profile_reset(void);
uint32_t decode_profile_attn_cycles(void);
uint32_t decode_profile_ffn_cycles(void);
void fwd_head_into(Endpoint endpoint, Matrix token_ids, Matrix out);
Matrix layers_prefill_fwd(SmolLMLayerShard layers, Matrix x);
void fwd_tail_last_into(Endpoint endpoint, Matrix x, Matrix logits_out);

#endif

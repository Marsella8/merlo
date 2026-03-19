#include "load.h"

static inline size_t f32_matrix_bytes(size_t rows, size_t cols) {
    return rows * cols * sizeof(float);
}

static size_t embed_tokens_bytes(void) {
    return qnum_bytes_for_shape(HIDDEN_SIZE, VOCAB_SIZE);
}

static inline size_t attn_norm_bytes(void) {
    return f32_matrix_bytes(1, HIDDEN_SIZE);
}

static inline size_t q_proj_bytes(void) {
    return qnum_bytes_for_shape(HIDDEN_SIZE, HIDDEN_SIZE);
}

static inline size_t k_proj_bytes(void) {
    return qnum_bytes_for_shape(HIDDEN_SIZE, KV_SIZE);
}

static inline size_t v_proj_bytes(void) {
    return qnum_bytes_for_shape(HIDDEN_SIZE, KV_SIZE);
}

static inline size_t o_proj_bytes(void) {
    return qnum_bytes_for_shape(HIDDEN_SIZE, HIDDEN_SIZE);
}

static inline size_t ffn_norm_bytes(void) {
    return f32_matrix_bytes(1, HIDDEN_SIZE);
}

static inline size_t gate_proj_bytes(void) {
    return qnum_bytes_for_shape(HIDDEN_SIZE, INTERMEDIATE_SIZE);
}

static inline size_t up_proj_bytes(void) {
    return qnum_bytes_for_shape(HIDDEN_SIZE, INTERMEDIATE_SIZE);
}

static inline size_t down_proj_bytes(void) {
    return qnum_bytes_for_shape(INTERMEDIATE_SIZE, HIDDEN_SIZE);
}

static inline size_t layer_attn_norm_offset(void) {
    return 0;
}

static inline size_t layer_q_proj_offset(void) {
    return layer_attn_norm_offset() + attn_norm_bytes();
}

static inline size_t layer_k_proj_offset(void) {
    return layer_q_proj_offset() + q_proj_bytes();
}

static inline size_t layer_v_proj_offset(void) {
    return layer_k_proj_offset() + k_proj_bytes();
}

static inline size_t layer_o_proj_offset(void) {
    return layer_v_proj_offset() + v_proj_bytes();
}

static inline size_t layer_ffn_norm_offset(void) {
    return layer_o_proj_offset() + o_proj_bytes();
}

static inline size_t layer_gate_proj_offset(void) {
    return layer_ffn_norm_offset() + ffn_norm_bytes();
}

static inline size_t layer_up_proj_offset(void) {
    return layer_gate_proj_offset() + gate_proj_bytes();
}

static inline size_t layer_down_proj_offset(void) {
    return layer_up_proj_offset() + up_proj_bytes();
}

static size_t layer_bytes(void) {
    return layer_down_proj_offset() + down_proj_bytes();
}

static Matrix load_matrix(uint8_t *ptr, size_t rows, size_t cols) {
    return (Matrix){
        .buffer = watch_buf(ptr, f32_matrix_bytes(rows, cols)),
        .rows = rows,
        .cols = cols,
        .row_stride = (int)cols,
        .col_stride = 1,
        .offset = 0,
        .owned = true,
    };
}

static QMatrix load_qmatrix(uint8_t *ptr, size_t rows, size_t cols, size_t size_bytes) {
    return qmat(watch_buf(ptr, size_bytes), rows, cols);
}

static Matrix load_attn_norm(uint8_t *ptr) {
    return load_matrix(ptr, 1, HIDDEN_SIZE);
}

static QMatrix load_q_proj(uint8_t *ptr) {
    return load_qmatrix(ptr, HIDDEN_SIZE, HIDDEN_SIZE, q_proj_bytes());
}

static QMatrix load_k_proj(uint8_t *ptr) {
    return load_qmatrix(ptr, KV_SIZE, HIDDEN_SIZE, k_proj_bytes());
}

static QMatrix load_v_proj(uint8_t *ptr) {
    return load_qmatrix(ptr, KV_SIZE, HIDDEN_SIZE, v_proj_bytes());
}

static QMatrix load_o_proj(uint8_t *ptr) {
    return load_qmatrix(ptr, HIDDEN_SIZE, HIDDEN_SIZE, o_proj_bytes());
}

static Matrix load_ffn_norm(uint8_t *ptr) {
    return load_matrix(ptr, 1, HIDDEN_SIZE);
}

static QMatrix load_gate_proj(uint8_t *ptr) {
    return load_qmatrix(ptr, INTERMEDIATE_SIZE, HIDDEN_SIZE, gate_proj_bytes());
}

static QMatrix load_up_proj(uint8_t *ptr) {
    return load_qmatrix(ptr, INTERMEDIATE_SIZE, HIDDEN_SIZE, up_proj_bytes());
}

static QMatrix load_down_proj(uint8_t *ptr) {
    return load_qmatrix(ptr, HIDDEN_SIZE, INTERMEDIATE_SIZE, down_proj_bytes());
}

static Matrix load_out_norm(uint8_t *ptr) {
    return load_matrix(ptr, 1, HIDDEN_SIZE);
}

static QMatrix load_lm_head(uint8_t *ptr) {
    return load_qmatrix(ptr, VOCAB_SIZE, HIDDEN_SIZE, embed_tokens_bytes());
}

static LayerCache init_layer_cache(void) {
    return (LayerCache){
        .k = empty(KV_SIZE, MAX_SEQ_LEN),
        .v = empty(KV_SIZE, MAX_SEQ_LEN),
    };
}

Block load_block_at(uint8_t *layer_ptr) {
    return (Block){
        .attn_norm = load_attn_norm(layer_ptr + layer_attn_norm_offset()),
        .q = load_q_proj(layer_ptr + layer_q_proj_offset()),
        .k = load_k_proj(layer_ptr + layer_k_proj_offset()),
        .v = load_v_proj(layer_ptr + layer_v_proj_offset()),
        .o = load_o_proj(layer_ptr + layer_o_proj_offset()),
        .ffn_norm = load_ffn_norm(layer_ptr + layer_ffn_norm_offset()),
        .gate = load_gate_proj(layer_ptr + layer_gate_proj_offset()),
        .up = load_up_proj(layer_ptr + layer_up_proj_offset()),
        .down = load_down_proj(layer_ptr + layer_down_proj_offset()),
        .cache = init_layer_cache(),
    };
}

Endpoint load_endpoint_at(uint8_t *embed_ptr, uint8_t *out_norm_ptr) {
    return (Endpoint){
        .out_norm = load_out_norm(out_norm_ptr),
        .lm_head = load_lm_head(embed_ptr),
    };
}

SmolLMLayerShard load_layer_shard_at(uint8_t *layer_ptr, size_t num_layers) {
    assert(num_layers <= MAX_NUM_LAYERS);

    SmolLMLayerShard layers = {
        .num_layers = num_layers,
    };

    for (size_t i = 0; i < num_layers; i++) {
        layers.blocks[i] = load_block_at(layer_ptr);
        layer_ptr += layer_bytes();
    }

    return layers;
}

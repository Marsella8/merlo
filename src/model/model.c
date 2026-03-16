#include "model.h"
#include "nn.h"
#include "utils.h"

typedef Matrix (*AttnFn)(Matrix, Matrix, Matrix, Matrix, Matrix, LayerCache, size_t);

static LayerCache init_layer_cache();

Matrix load_matrix(const char* name, const int layer) {
    /*
    const char* path = "weights/";
    char dest[256];
    if (layer == -1) {
        snprintf(dest, 256, "%s%s", path, name);
    } else {
        snprintf(dest, 256, "%s%d.%s", path, layer, name);
    }
    FILE* file = fopen(dest, "rb");
    assert(file != NULL);

    size_t row, col;
    size_t row_read = fread(&row, sizeof(size_t), 1, file);
    size_t col_read = fread(&col, sizeof(size_t), 1, file);
#ifdef SAFETY
    assert(row_read == 1);
    assert(col_read == 1);
#endif

    size_t numel = row * col;
    Buffer* buffer = buf(numel * sizeof(float));
    fread(buffer->data, sizeof(float), numel, file);
    fclose(file);
    return mat(buffer, row, col);
    */
    (void)name;
    (void)layer;
    not_implemented();
    return empty(0, 0);
}

QMatrix load_qmatrix(const char *name, const int layer) {
    /*
    const char* path = "weights/";
    char dest[256];
    if (layer == -1) {
        snprintf(dest, 256, "%s%s", path, name);
    } else {
        snprintf(dest, 256, "%s%d.%s", path, layer, name);
    }
    FILE* file = fopen(dest, "rb");
    assert(file != NULL);

    size_t row, col;
    size_t row_read = fread(&row, sizeof(size_t), 1, file);
    size_t col_read = fread(&col, sizeof(size_t), 1, file);
#ifdef SAFETY
    assert(row_read == 1);
    assert(col_read == 1);
    assert(row % 32 == 0);
#endif

    size_t numel = row * col;
    size_t num_blocks = numel / 32;

    Buffer* scales = buf(num_blocks * sizeof(float));
    size_t scales_read = fread(scales->data, sizeof(float), num_blocks, file);
#ifdef SAFETY
    assert(scales_read == num_blocks);
#endif

    Buffer* weights = buf(numel * sizeof(int8_t));
    size_t weights_read = fread(weights->data, sizeof(int8_t), numel, file);
#ifdef SAFETY
    assert(weights_read == numel);
#endif

    fclose(file);
    return qmat(weights, scales, row, col);
    */
    (void)name;
    (void)layer;
    not_implemented();
    return qmat(NULL, NULL, 0, 0);
}

static Block load_block(size_t layer) {
    return (Block) {
        .attn_norm = load_matrix("attn_norm", layer),
        .q = load_matrix("attn_q", layer),
        .k = load_matrix("attn_k", layer),
        .v = load_matrix("attn_v", layer),
        .o = load_matrix("attn_output", layer),
        .ffn_norm = load_matrix("ffn_norm", layer),
        .gate = load_matrix("ffn_gate", layer),
        .up = load_matrix("ffn_up", layer),
        .down = load_matrix("ffn_down", layer),
        .cache = init_layer_cache(),
    };
}

static LayerCache init_layer_cache() {
    return (LayerCache){
        .k = empty(KV_SIZE, MAX_SEQ_LEN),
        .v = empty(KV_SIZE, MAX_SEQ_LEN),
    };
}

SmolLMHeadShard load_head_shard() {
    Matrix embd_transposed = load_matrix("token_embd", -1);
    Matrix emb = as_contiguous(transpose(embd_transposed));
    free_mat(embd_transposed);
    return (SmolLMHeadShard){ .embeddings = emb };
}

SmolLMLayerShard load_layer_shard(size_t start_layer, size_t end_layer) {
    SmolLMLayerShard layers = {0};
    layers.num_layers = end_layer - start_layer;
    layers.blocks = malloc(layers.num_layers * sizeof(Block));
    for (size_t i = 0; i < layers.num_layers; i++) {
        layers.blocks[i] = load_block(start_layer + i);
    }
    return layers;
}

SmolLMTailShard load_tail_shard() {
    SmolLMTailShard tail = {0};
    tail.final_norm = load_matrix("output_norm", -1);
    tail.lm_head = load_matrix("token_embd", -1);
    return tail;
}

SmolLM2 load_model(size_t num_layers) {
    return (SmolLM2){
        .head = load_head_shard(),
        .layers = load_layer_shard(0, num_layers),
        .tail = load_tail_shard(),
    };
}

SmolLM2 load_main_model() {
    return load_model(NUM_MAIN_LAYERS);
}

SmolLM2 load_spec_model() {
    return load_model(NUM_SPEC_LAYERS);
}

static void free_block(Block b) {
    Matrix mats[] = { b.attn_norm, b.q, b.k, b.v, b.o, b.ffn_norm, b.gate, b.up, b.down, b.cache.k, b.cache.v };
    for (size_t i = 0; i < sizeof(mats) / sizeof(mats[0]); i++)
        free_mat(mats[i]);
}

void free_head_shard(SmolLMHeadShard head) {
    free_mat(head.embeddings);
}

void free_layer_shard(SmolLMLayerShard layers) {
    for (size_t i = 0; i < layers.num_layers; i++) {
        free_block(layers.blocks[i]);
    }
    free(layers.blocks);
}

void free_tail_shard(SmolLMTailShard tail) {
    free_mat(tail.final_norm);
    free_mat(tail.lm_head);
}

void free_model(SmolLM2 model) {
    free_head_shard(model.head);
    free_layer_shard(model.layers);
    free_tail_shard(model.tail);
}

static Matrix layer_fwd(Block b, Matrix x, size_t pos, AttnFn attn_fwd) {
    Matrix attn_norm = rms_norm(x, b.attn_norm);
    Matrix attn = attn_fwd(attn_norm, b.q, b.k, b.v, b.o, b.cache, pos);
    free_mat(attn_norm);
    Matrix attn_out = add(x, attn);
    free_mat(attn);

    Matrix ffn_norm = rms_norm(attn_out, b.ffn_norm);
    Matrix ffnet = ffn(ffn_norm, b.gate, b.up, b.down);
    free_mat(ffn_norm);
    Matrix ffn_out = add(attn_out, ffnet);
    free_mat(attn_out);
    free_mat(ffnet);

    return ffn_out;
}

Matrix head_fwd(SmolLMHeadShard head, Matrix token_ids) {
    return embed(head.embeddings, token_ids);
}

static Matrix prefill_from_zero(Matrix x, Matrix Wq, Matrix Wk, Matrix Wv, Matrix Wo, LayerCache cache, size_t pos) {
    return prefill_gqa(x, Wq, Wk, Wv, Wo, cache);
}

Matrix layers_prefill_fwd(SmolLMLayerShard layers, Matrix x) {
#ifdef SAFETY
    assert(layers.num_layers > 0);
#endif
    Matrix out = layer_fwd(layers.blocks[0], x, 0, prefill_from_zero);
    for (size_t l = 1; l < layers.num_layers; l++) {
        Matrix next = layer_fwd(layers.blocks[l], out, 0, prefill_from_zero);
        free_mat(out);
        out = next;
    }
    return out;
}

Matrix layers_decode_fwd(SmolLMLayerShard layers, Matrix x, size_t pos) {
#ifdef SAFETY
    assert(layers.num_layers > 0);
#endif
    Matrix out = layer_fwd(layers.blocks[0], x, pos, decode_gqa);
    for (size_t l = 1; l < layers.num_layers; l++) {
        Matrix next = layer_fwd(layers.blocks[l], out, pos, decode_gqa);
        free_mat(out);
        out = next;
    }
    return out;
}

Matrix tail_fwd(SmolLMTailShard tail, Matrix x) {
    Matrix normed = rms_norm(x, tail.final_norm);
    Matrix logits = matmul(normed, tail.lm_head);
    free_mat(normed);
    return logits;
}

void prefill(SmolLM2 model, Matrix x) {
    Matrix head_out = head_fwd(model.head, x);
    Matrix layered = layers_prefill_fwd(model.layers, head_out);
    free_mat(head_out);
    free_mat(layered);
}

Matrix fwd(SmolLM2 model, size_t token_id, size_t pos) {
    Matrix token = empty(1, 1);
    *at(token, 0, 0) = (float)token_id;

    Matrix head_out = head_fwd(model.head, token);
    free_mat(token);

    Matrix layered = layers_decode_fwd(model.layers, head_out, pos);
    free_mat(head_out);
    Matrix logits = tail_fwd(model.tail, layered);
    free_mat(layered);
    return logits;
}

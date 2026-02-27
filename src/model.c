#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "matrix.h"
#include "model.h"
#include "nn.h"
#include "utils.h"
#include <stdint.h>

const char* PATH = "model/";
typedef Matrix (*AttnFn)(Matrix, Matrix, Matrix, Matrix, Matrix, LayerCache, size_t);

static FILE* get_file(const char *name, const int layer) {
    char dest[256]; 
    if (layer == -1) {
        snprintf(dest, 256, "%s%s", PATH, name);
    } else {
        snprintf(dest, 256, "%s%d.%s", PATH, layer, name);
    }
    FILE* f = fopen(dest, "rb");
    assert(f!=NULL);
    return f;
}

Matrix load_matrix(const char* name, const int layer) {
    FILE* file =  get_file(name, layer);
    
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
}

QMatrix load_qmatrix(const char *name, const int layer) {
    FILE* file =  get_file(name, layer);
    
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
    };
}

static KVCache init_kvcache(size_t num_layers) {
    KVCache cache = {0};
    cache.caches = malloc(num_layers * sizeof(LayerCache));
    cache.size = num_layers;
    for (size_t i = 0; i < num_layers; i++) {
        cache.caches[i].k = empty(KV_SIZE, MAX_SEQ_LEN);
        cache.caches[i].v = empty(KV_SIZE, MAX_SEQ_LEN);
    }
    return cache;
}

SmolLM2 load_model(size_t num_layers) {
    SmolLM2 model;
    model.num_layers = num_layers;

    Matrix embd_transposed = load_matrix("token_embd", -1);
    model.embeddings = transpose(embd_transposed);
    model.blocks = malloc(num_layers * sizeof(Block));
    for (size_t i = 0; i < num_layers; i++) {
        model.blocks[i] = load_block(i);
    }

    model.final_norm = load_matrix("output_norm", -1);
    model.lm_head = embd_transposed;

    model.cache = init_kvcache(num_layers);

    return model;
}

SmolLM2 load_main_model() {
    return load_model(NUM_MAIN_LAYERS);
}

SmolLM2 load_spec_model() {
    return load_model(NUM_SPEC_LAYERS);
}

static void free_block(Block b) {
    free_mat(b.attn_norm);
    free_mat(b.q);
    free_mat(b.k);
    free_mat(b.v);
    free_mat(b.o);
    free_mat(b.ffn_norm);
    free_mat(b.gate);
    free_mat(b.up);
    free_mat(b.down);
}

void free_model(SmolLM2 model) {
    free_mat(model.lm_head);
    for (size_t i = 0; i < model.num_layers; i++) {
        free_block(model.blocks[i]);
    }
    free(model.blocks);
    free_mat(model.final_norm);
    free_kvcache(model.cache);
}

void free_kvcache(KVCache cache) {
    for (size_t i = 0; i < cache.size; i++) {
        free_mat(cache.caches[i].k);
        free_mat(cache.caches[i].v);
    }
    free(cache.caches);
}

static Matrix layer_fwd(Block b, Matrix x, LayerCache cache, size_t pos, AttnFn attn_fwd) {
    Matrix attn_norm = rms_norm(x, b.attn_norm);
    Matrix attn = attn_fwd(attn_norm, b.q, b.k, b.v, b.o, cache, pos);
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

static Matrix prefill_from_zero(Matrix x, Matrix Wq, Matrix Wk, Matrix Wv, Matrix Wo, LayerCache cache, size_t pos) {
    (void)pos;
    return prefill_gqa(x, Wq, Wk, Wv, Wo, cache);
}

void prefill(SmolLM2 model, Matrix x) {
    Matrix out = embed(model.embeddings, x);
    for (size_t l = 0; l < model.num_layers; l++) {
        Matrix next = layer_fwd(model.blocks[l], out, model.cache.caches[l], 0, prefill_from_zero);
        free_mat(out);
        out = next;
    }
    free_mat(out);
}

Matrix fwd(SmolLM2 model, size_t token_id, size_t pos) {
    Matrix token = empty(1, 1);
    *at(token, 0, 0) = (float)token_id;

    Matrix out = embed(model.embeddings, token);
    free_mat(token);

    for (size_t l = 0; l < model.num_layers; l++) {
        Matrix next = layer_fwd(model.blocks[l], out, model.cache.caches[l], pos, decode_gqa);
        free_mat(out);
        out = next;
    }

    Matrix normed = rms_norm(out, model.final_norm);
    Matrix logits = matmul(normed, model.lm_head);
    free_mat(out);
    free_mat(normed);
    return logits;
}

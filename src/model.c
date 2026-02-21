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
    fread(&row, sizeof(size_t), 1, file);
    fread(&col, sizeof(size_t), 1, file);
    
    size_t numel = row * col;
    Buffer* buffer = buf(numel * sizeof(float));
    fread(buffer->data, sizeof(float), numel, file);
    fclose(file);
    
    return mat(buffer, row, col);
}

QMatrix load_qmatrix(const char *name, const int layer) {
    FILE* file =  get_file(name, layer);
    
    size_t row, col;
    fread(&row, sizeof(size_t), 1, file);
    fread(&col, sizeof(size_t), 1, file);
    
    size_t numel = row * col;
    size_t num_blocks = numel / 32;
    
    Buffer* scales = buf(num_blocks * sizeof(float));
    fread(scales->data, sizeof(float), num_blocks, file);
    
    Buffer* weights = buf(numel * sizeof(int8_t));
    fread(weights->data, sizeof(int8_t), numel, file);
    
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

static KVCache init_kvcache() {
    KVCache cache = {0};
    for (size_t i = 0; i < NUM_LAYERS; i++) {
        cache.caches[i].k = empty(KV_SIZE, MAX_SEQ_LEN);
        cache.caches[i].v = empty(KV_SIZE, MAX_SEQ_LEN);
    }
    return cache;
}

SmolLM2 load_model() {
    SmolLM2 model;

    Matrix embd_transposed = load_matrix("token_embd", -1);
    model.embeddings = transpose(embd_transposed);

    for (size_t i = 0; i < NUM_LAYERS; i++) {
        model.blocks[i] = load_block(i);
    }

    model.final_norm = load_matrix("output_norm", -1);
    model.lm_head = embd_transposed;

    model.cache = init_kvcache();

    return model;
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
    for (int i = 0; i < NUM_LAYERS; i++) {
        free_block(model.blocks[i]);
    }
    free_mat(model.final_norm);
    free_kvcache(model.cache);
}

void free_kvcache(KVCache cache) {
    for (int i = 0; i < NUM_LAYERS; i++) {
        free_mat(cache.caches[i].k);
        free_mat(cache.caches[i].v);
    }
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

void prefill(SmolLM2 model, Matrix x, size_t pos) {
    Matrix out = embed(model.embeddings, x);
    for (int l = 0; l < NUM_LAYERS; l++) {
        Matrix next = layer_fwd(model.blocks[l], out, model.cache.caches[l], pos, prefill_gqa);
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

    for (int l = 0; l < NUM_LAYERS; l++) {
        Matrix next = layer_fwd(model.blocks[l], out, model.cache.caches[l], pos, prefill_gqa);
        free_mat(out);
        out = next;
    }

    Matrix normed = rms_norm(out, model.final_norm);
    Matrix logits = matmul(normed, model.lm_head);
    free_mat(out);
    free_mat(normed);
    return logits;
}

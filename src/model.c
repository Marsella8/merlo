#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "matrix.h"
#include "model.h"
#include <stdint.h>

const char* PATH = "model/";

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

static void free_block(Block* b) {
    free_mat(b->attn_norm);
    free_mat(b->q);
    free_mat(b->k);
    free_mat(b->v);
    free_mat(b->o);
    free_mat(b->ffn_norm);
    free_mat(b->gate);
    free_mat(b->up);
    free_mat(b->down);
}

size_t size(LayerCache cache) {
    #ifdef SAFETY
        assert(cache.k.rows == KV_SIZE);
        assert(cache.v.rows == KV_SIZE);
        assert(cache.k.cols == cache.v.cols);
    #endif
        return cache.k.cols;
    }
    
    void free_model(SmolLM2 model) {
        free_mat(model.embeddings);
        for (int i = 0; i < NUM_LAYERS; i++) {
            free_block(&model.blocks[i]);
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
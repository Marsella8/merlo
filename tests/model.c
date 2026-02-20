#include <assert.h>
#include <stdio.h>
#include "model.h"
#include "matrix.h"

void test_embeddings_shape() {
    printf("  test_embeddings_shape\n");
    SmolLM2 model = load_model();
    
    assert(model.embeddings.rows == VOCAB_SIZE);
    assert(model.embeddings.cols == HIDDEN_SIZE);
    
}

void test_block_shapes() {
    printf("  test_block_shapes\n");
    SmolLM2 model = load_model();
    
    for (size_t i = 0; i < NUM_LAYERS; i++) {
        Block block = model.blocks[i];
        
        assert(block.attn_norm.rows == 1);
        assert(block.attn_norm.cols == HIDDEN_SIZE);
        
        assert(block.q.rows == HIDDEN_SIZE);
        assert(block.q.cols == HIDDEN_SIZE);
        
        assert(block.k.rows == HIDDEN_SIZE);
        assert(block.k.cols == HEAD_DIM * NUM_KV_HEADS);
        
        assert(block.v.rows == HIDDEN_SIZE);
        assert(block.v.cols == HEAD_DIM * NUM_KV_HEADS);
        
        assert(block.o.rows == HIDDEN_SIZE);
        assert(block.o.cols == HIDDEN_SIZE);
        
        assert(block.ffn_norm.rows == 1);
        assert(block.ffn_norm.cols == HIDDEN_SIZE);
        
        assert(block.gate.rows == HIDDEN_SIZE);
        assert(block.gate.cols == INTERMEDIATE_SIZE);
        
        assert(block.up.rows == HIDDEN_SIZE);
        assert(block.up.cols == INTERMEDIATE_SIZE);
        
        assert(block.down.rows == INTERMEDIATE_SIZE);
        assert(block.down.cols == HIDDEN_SIZE);
    }
}

void test_final_norm_shape() {
    printf("  test_final_norm_shape\n");
    SmolLM2 model = load_model();
    
    assert(model.final_norm.rows == 1);
    assert(model.final_norm.cols == HIDDEN_SIZE);
}

void test_lm_head_shape() {
    printf("  test_lm_head_shape\n");
    SmolLM2 model = load_model();
    
    assert(model.lm_head.rows == HIDDEN_SIZE);
    assert(model.lm_head.cols == VOCAB_SIZE);
}

int main() {
    printf("model tests:\n");
    test_embeddings_shape();
    test_block_shapes();
    test_final_norm_shape();
    test_lm_head_shape();
    return 0;
}


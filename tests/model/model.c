#include <assert.h>
#include <stdio.h>

#include "model.h"
#include "tokenizer.h"

void test_embeddings_shape() {
    printf("  test_embeddings_shape\n");
    SmolLM2 model = load_main_model();
    
    assert(model.head.embeddings.rows == VOCAB_SIZE);
    assert(model.head.embeddings.cols == HIDDEN_SIZE);
    free_model(model);
}

void test_block_shapes() {
    printf("  test_block_shapes\n");
    SmolLM2 model = load_main_model();
    
    for (size_t i = 0; i < model.layers.num_layers; i++) {
        Block block = model.layers.blocks[i];
        
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
    free_model(model);
}

void test_final_norm_shape() {
    printf("  test_final_norm_shape\n");
    SmolLM2 model = load_main_model();
    
    assert(model.tail.final_norm.rows == 1);
    assert(model.tail.final_norm.cols == HIDDEN_SIZE);
    free_model(model);
}

void test_load_layer_shard_range() {
    printf("  test_load_layer_shard_range\n");
    SmolLMLayerShard layers = load_layer_shard(2, 5);

    assert(layers.num_layers == 3);
    for (size_t i = 0; i < layers.num_layers; i++) {
        assert(layers.blocks[i].attn_norm.rows == 1);
        assert(layers.blocks[i].attn_norm.cols == HIDDEN_SIZE);
    }

    free_layer_shard(layers);
}

void test_lm_head_shape() {
    printf("  test_lm_head_shape\n");
    SmolLM2 model = load_main_model();
    
    assert(model.tail.lm_head.rows == HIDDEN_SIZE);
    assert(model.tail.lm_head.cols == VOCAB_SIZE);
    free_model(model);
}

void test_prefill() {
    printf("  test_prefill\n");
    SmolLM2 model = load_main_model();
    Vocab vocab = load_vocab();

    const char* prompt = "Hello my name is";
    Matrix tokens = tokenize(prompt, vocab);

    prefill(model, tokens);

    free_mat(tokens);
    free_model(model);
    free_vocab(vocab);
}

void test_decode() {
    printf("  test_decode\n");
    SmolLM2 model = load_main_model();
    Vocab vocab = load_vocab();

    const char* prompt = "Hello my name is";
    Matrix tokens = tokenize(prompt, vocab);
    prefill(model, tokens);

    size_t decode_pos = tokens.cols;
    size_t token_id = (size_t)*at(tokens, 0, tokens.cols - 1);
    Matrix logits = fwd(model, token_id, decode_pos);
    assert(logits.rows == 1);
    assert(logits.cols == VOCAB_SIZE);

    free_mat(logits);
    free_mat(tokens);
    free_model(model);
    free_vocab(vocab);
}

int main() {
    printf("model tests:\n");
    test_embeddings_shape();
    test_block_shapes();
    test_final_norm_shape();
    test_load_layer_shard_range();
    test_lm_head_shape();
    test_prefill();
    test_decode();
    return 0;
}

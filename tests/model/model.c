#include <stdint.h>

#include "load.h"
#include "model.h"

#define FAKE_PTR(addr) ((uint8_t *)(uintptr_t)(addr))

static void test_endpoint_shapes(void) {
    printk("  test_endpoint_shapes\n");
    Endpoint endpoint = load_endpoint_at(FAKE_PTR(0x100000), FAKE_PTR(0x200000));

    assert(endpoint.out_norm.rows == 1);
    assert(endpoint.out_norm.cols == HIDDEN_SIZE);
    assert(endpoint.lm_head.rows == HIDDEN_SIZE);
    assert(endpoint.lm_head.cols == VOCAB_SIZE);

}

static void test_block_shapes(void) {
    printk("  test_block_shapes\n");
    SmolLMLayerShard layers = load_layer_shard_at(FAKE_PTR(0x300000), 1);
    Block block = layers.blocks[0];

    assert(block.attn_norm.rows == 1);
    assert(block.attn_norm.cols == HIDDEN_SIZE);
    assert(block.q.rows == HIDDEN_SIZE);
    assert(block.q.cols == HIDDEN_SIZE);
    assert(block.k.rows == HIDDEN_SIZE);
    assert(block.k.cols == KV_SIZE);
    assert(block.v.rows == HIDDEN_SIZE);
    assert(block.v.cols == KV_SIZE);
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

void notmain(void) {
    kmalloc_init();
    printk("model tests:\n");
    test_endpoint_shapes();
    test_block_shapes();
    printk("tests/single/model/model: pass\n");
    clean_reboot();
}

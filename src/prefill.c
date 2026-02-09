#include "prefill.h"
#include "model.h"
#include "nn.h"

Matrix prefill_layer_fwd(Block b, Matrix x, int pos, LayerCache* cache) {
    Matrix attn_norm = rms_norm(x, b.attn_norm);
    Matrix attn = prefill_gqa(attn_norm, b.q, b.k, b.v, b.o, pos, cache);
    free_buf(attn_norm.buffer);
    Matrix attn_out = add(x, attn);
    free_buf(attn.buffer);

    Matrix ffn_norm = rms_norm(attn_out, b.ffn_norm);
    Matrix ffnet = ffn(ffn_norm, b.gate, b.up, b.down);
    free_buf(ffn_norm.buffer);
    Matrix ffn_out = add(attn_out, ffnet);
    free_buf(attn_out.buffer);
    free_buf(ffnet.buffer);

    return ffn_out;
}


KVCache prefill(SmolLM2 model, Matrix x) {
    KVCache cache;
    Matrix out = embed(model.embeddings, x);
    for (int l = 0; l < NUM_LAYERS; l++) {
        out = prefill_layer_fwd(model.blocks[l], out, 0, &cache.caches[l]);
    }
    return cache;
}
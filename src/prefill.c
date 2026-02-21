#include "prefill.h"
#include "model.h"
#include "nn.h"

Matrix prefill_layer_fwd(Block b, Matrix x, LayerCache cache) {
    Matrix attn_norm = rms_norm(x, b.attn_norm);
    Matrix attn = prefill_gqa(attn_norm, b.q, b.k, b.v, b.o, cache);
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


void prefill(SmolLM2* model, Matrix x) {
    resize_kv(model, x.cols);
    Matrix out = embed(model->embeddings, x);
    for (int l = 0; l < NUM_LAYERS; l++) {
        out = prefill_layer_fwd(model->blocks[l], out, model->cache.caches[l]);
    }
}

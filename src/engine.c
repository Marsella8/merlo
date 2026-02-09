#include "engine.h"
#include "model.h"
#include "nn.h"
#include "sample.h"
#include "utils.h"

Matrix engine_layer_fwd(Block b, Matrix x, int pos, LayerCache* cache) {
    Matrix attn_norm = rms_norm(x, b.attn_norm);
    Matrix attn = engine_gqa(attn_norm, b.q, b.k, b.v, b.o, cache, pos);
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


size_t fwd(SmolLM2 model, KVCache* cache, size_t token_id, size_t pos) {
    not_implemented();
    return 0;
}
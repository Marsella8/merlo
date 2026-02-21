#include "decode.h"
#include "model.h"
#include "nn.h"
#include "sample.h"
#include "utils.h"

Matrix decode_layer_fwd(Block b, Matrix x, LayerCache cache) {
    Matrix attn_norm = rms_norm(x, b.attn_norm);
    Matrix attn = decode_gqa(attn_norm, b.q, b.k, b.v, b.o, cache);
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


Matrix fwd(SmolLM2* model, size_t token_id) {
    increment_kv(model);
    not_implemented();
    Matrix m = {0};
    return m;
}
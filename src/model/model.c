#include "model.h"
#include "nn.h"

Matrix block_prefill_fwd(Block b, Matrix x) {
#ifdef SAFETY
    assume_shape(x, -1, HIDDEN_SIZE);
#endif
    Matrix attn_norm = empty(x.rows, HIDDEN_SIZE);
    rms_norm_into(x, b.attn_norm, attn_norm);
    Matrix attn = prefill_gqa(attn_norm, b.q, b.k, b.v, b.o, b.cache);
    free_mat(attn_norm);

    Matrix attn_out = empty(x.rows, HIDDEN_SIZE);
    add_into(x, attn, attn_out);
    free_mat(attn);

    Matrix ffn_norm = empty(x.rows, HIDDEN_SIZE);
    rms_norm_into(attn_out, b.ffn_norm, ffn_norm);
    Matrix ffnet = empty(x.rows, HIDDEN_SIZE);
    ffn_prefill_into(ffn_norm, b.gate, b.up, b.down, ffnet);
    free_mat(ffn_norm);

    Matrix ffn_out = empty(x.rows, HIDDEN_SIZE);
    add_into(attn_out, ffnet, ffn_out);
    free_mat(attn_out);
    free_mat(ffnet);

    return ffn_out;
}

void block_decode_fwd_into(Block b, Matrix x, size_t pos, Matrix out) {
#ifdef SAFETY
    assume_shape(x, 1, HIDDEN_SIZE);
    assert(out.rows == 1 && out.cols == HIDDEN_SIZE);
#endif
    float attn_norm_data[HIDDEN_SIZE];
    float attn_data[HIDDEN_SIZE];
    float attn_out_data[HIDDEN_SIZE];
    float ffn_norm_data[HIDDEN_SIZE];
    float ffnet_data[HIDDEN_SIZE];
    Buffer attn_norm_buf;
    Buffer attn_buf;
    Buffer attn_out_buf;
    Buffer ffn_norm_buf;
    Buffer ffnet_buf;
    Matrix attn_norm = stack_mat(&attn_norm_buf, attn_norm_data, 1, HIDDEN_SIZE);
    Matrix attn = stack_mat(&attn_buf, attn_data, 1, HIDDEN_SIZE);
    Matrix attn_out = stack_mat(&attn_out_buf, attn_out_data, 1, HIDDEN_SIZE);
    Matrix ffn_norm = stack_mat(&ffn_norm_buf, ffn_norm_data, 1, HIDDEN_SIZE);
    Matrix ffnet = stack_mat(&ffnet_buf, ffnet_data, 1, HIDDEN_SIZE);

    rms_norm_into(x, b.attn_norm, attn_norm);
    decode_gqa_into(attn_norm, b.q, b.k, b.v, b.o, b.cache, pos, attn);
    add_into(x, attn, attn_out);

    rms_norm_into(attn_out, b.ffn_norm, ffn_norm);
    ffn_decode_into(ffn_norm, b.gate, b.up, b.down, ffnet);
    add_into(attn_out, ffnet, out);
}

void fwd_head_into(Endpoint endpoint, Matrix token_ids, Matrix out) {
    embed_into(endpoint.lm_head, token_ids, out);
}

Matrix layers_prefill_fwd(SmolLMLayerShard layers, Matrix x) {
#ifdef SAFETY
    assert(layers.num_layers > 0);
#endif
    Matrix out = block_prefill_fwd(layers.blocks[0], x);
    for (size_t l = 1; l < layers.num_layers; l++) {
        Matrix next = block_prefill_fwd(layers.blocks[l], out);
        free_mat(out);
        out = next;
    }
    return out;
}

void fwd_tail_last_into(Endpoint endpoint, Matrix x, Matrix logits_out) {
#ifdef SAFETY
    assert(x.rows > 0);
    assume_shape(x, -1, HIDDEN_SIZE);
    assume_shape(endpoint.out_norm, 1, HIDDEN_SIZE);
    assume_shape(endpoint.lm_head, VOCAB_SIZE, HIDDEN_SIZE);
    assume_shape(logits_out, 1, VOCAB_SIZE);
#endif
    float normed_data[HIDDEN_SIZE];
    Buffer normed_buf;
    Matrix last_hidden = slice(x, x.rows - 1, x.rows, 0, x.cols);
    Matrix normed = stack_mat(&normed_buf, normed_data, 1, HIDDEN_SIZE);

    rms_norm_into(last_hidden, endpoint.out_norm, normed);
    qmatvec_into(normed, endpoint.lm_head, logits_out);
}

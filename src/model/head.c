// The head owns the embedding/tail ends of the pipeline.
#include <stdbool.h>

#include "comm.h"
#include "head.h"
#include "model.h"
#include "rpi.h"
#include "sample.h"
#include "tokenizer.h"
#include "utils.h"

static size_t sample_last_token(Matrix logits) {
    Matrix last_logits = slice(logits, logits.rows - 1, logits.rows, 0, logits.cols);
    return sample(last_logits, 0.9f);
}

static void send_embedded_tokens(SmolLMHeadShard head_shard, Matrix token_ids, size_t token_pos) {
    Matrix embedded = head_fwd(head_shard, token_ids);
    Buffer* buf = serialize_packet((Packet){
        .matrix = embedded,
        .token_pos = token_pos,
    });
    send(buf);
    free_buf(buf);
    free_mat(embedded);
}

int head() {
    const char* prompt = "Dogs are the most"; // TODO is the space prefix or suffix for thi
    size_t max_tokens = 100;

    comm_setup();

    SmolLMHeadShard head_shard = load_head_shard();
    SmolLMTailShard tail_shard = load_tail_shard();
    Vocab vocab = load_vocab();
    Matrix prompt_tokens = tokenize(prompt, vocab);
    size_t token_pos = prompt_tokens.cols;

    putk(prompt);
    send_embedded_tokens(head_shard, prompt_tokens, 0);
    free_mat(prompt_tokens);

    while (token_pos < max_tokens) {
        while (!string_queue_is_empty(string_queue)) {
            char* str = string_queue_deque(&string_queue);
            putk(str);
            free(str);
        }

        if (!packet_queue_is_empty(packet_queue)) {
            Packet pkt = packet_queue_deque(&packet_queue);
            Matrix logits = tail_fwd(tail_shard, pkt.matrix);
            size_t token = sample_last_token(logits);
            Matrix next_token = empty(1, 1);
            *at(next_token, 0, 0) = (float)token;

            putk(vocab.tokens[token]);

            free_mat(logits);
            free_mat(pkt.matrix);

            Matrix embedded = head_fwd(head_shard, next_token);
            Buffer* buf = serialize_packet((Packet){
                .matrix = embedded,
                .token_pos = token_pos,
            });
        
            send(buf);
            free_buf(buf);
            free_mat(embedded);
            free_mat(next_token);
            token_pos++;
        }
    }

    free_head_shard(head_shard);
    free_tail_shard(tail_shard);
    free_vocab(vocab);

    return 0;
}

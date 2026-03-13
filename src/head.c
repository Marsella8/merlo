// The head orchestrates, plus holds the head and tail shards, and the spec model
#include <stdbool.h>
#include <stdio.h>

#include "model.h"
#include "sample.h"
#include "tokenizer.h"
#include "comm.h"


// TODO(@pietro):
// head and shard should share the same matrix

int head() {
    SmolLMHeadShard head = load_head_shard();
    SmolLMTailShard tail = load_tail_shard();
    SmolLM2 spec_model = load_spec_model();
    Vocab vocab = load_vocab();
    size_t token_pos = 0;

    while (true) {

        // for printing things
        if (!string_queue_is_empty(string_queue)) {
            char* str = string_queue_deque(&string_queue);
            printf("[HEAD] %s\n", str);
        }

        // receives a token (TODO: compare against spec model, ...)
        if (!packet_queue_is_empty(packet_queue)) {
            Packet pkt = packet_queue_deque(&packet_queue);
            Matrix logits = tail_fwd(tail, pkt.matrix);
            size_t token = sample(logits, 0.9f);
            char* str = vocab.tokens[token];
            printf("%s", str);
            fflush(stdout);
            free_mat(logits);
            free_mat(pkt.matrix);
        }

        // schedule a new request according to spec model

        
    }

    return 0;
}
#include <stdbool.h>
#include <string.h>

#include "comm.h"
#include "load.h"
#include "model.h"
#include "rpi.h"
#include "sample.h"
#include "tokenizer.h"

#define HEAD_WEIGHTS_BASE ((uint8_t *)0x04000000)
#define HEAD_TX_PIN 20
#define HEAD_RX_PIN 21

static size_t head_embed_offset(void) {
    return VOCAB_SIZE * 256;
}

static size_t head_out_norm_offset(void) {
    return head_embed_offset() + qnum_bytes_for_shape(HIDDEN_SIZE, VOCAB_SIZE);
}

static size_t sample_last_token(Matrix logits) {
    Matrix last_logits = slice(logits, logits.rows - 1, logits.rows, 0, logits.cols);
    return sample(last_logits, 0.9f);
}

static void send_embedded_tokens(sw_uart_t *uart,
                                 Endpoint endpoint,
                                 Matrix token_ids,
                                 size_t token_pos) {
    Matrix embedded = empty(token_ids.cols, HIDDEN_SIZE);
    fwd_head_into(endpoint, token_ids, embedded);
    Buffer *buf = serialize_packet((Packet){
        .matrix = embedded,
        .token_pos = token_pos,
    });
    assert(comm_send(uart, buf));
    free_buf(buf);
    free_mat(embedded);
}

void notmain(void) {
    static const char *prompt = "The capital of France is";

    size_t max_tokens = 100;
    uint8_t *embed_ptr = HEAD_WEIGHTS_BASE + head_embed_offset();
    uint8_t *out_norm_ptr = HEAD_WEIGHTS_BASE + head_out_norm_offset();

    kmalloc_init();
    void enable_dcache(void);
    enable_dcache();
    sw_uart_t uart = comm_init(HEAD_TX_PIN, HEAD_RX_PIN);

    Endpoint endpoint = load_endpoint_at(embed_ptr, out_norm_ptr);
    Matrix prompt_tokens = tokenize(prompt);
    size_t token_pos = prompt_tokens.cols;

    send_embedded_tokens(&uart, endpoint, prompt_tokens, 0);
    free_mat(prompt_tokens);

    while (token_pos < max_tokens) {
        Buffer *frame = comm_recv(&uart);

        char *str = maybe_deserialize_string(frame);
        if (str != NULL) {
            putk(str);
            free(str);
            free_buf(frame);
            continue;
        }

        Packet pkt = maybe_deserialize_packet(frame);
        free_buf(frame);
        if (pkt.matrix.rows == 0 && pkt.matrix.cols == 0) {
            free_mat(pkt.matrix);
            continue;
        }

        float logits_data[VOCAB_SIZE];
        float next_token_data[1];
        float embedded_data[HIDDEN_SIZE];
        Buffer logits_buf;
        Buffer next_token_buf;
        Buffer embedded_buf;
        Matrix logits = stack_mat(&logits_buf, logits_data, 1, VOCAB_SIZE);
        Matrix next_token = stack_mat(&next_token_buf, next_token_data, 1, 1);
        Matrix embedded = stack_mat(&embedded_buf, embedded_data, 1, HIDDEN_SIZE);
        fwd_tail_last_into(endpoint, pkt.matrix, logits);

        size_t token = sample_last_token(logits);
        *at(next_token, 0, 0) = (float)token;

        print_token(token);

        free_mat(pkt.matrix);

        fwd_head_into(endpoint, next_token, embedded);
        Buffer *buf = serialize_packet((Packet){
            .matrix = embedded,
            .token_pos = token_pos,
        });
        assert(comm_send(&uart, buf));

        free_buf(buf);
        token_pos++;
    }
#undef HEAD_USER
}

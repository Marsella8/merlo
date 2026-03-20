#include <stdbool.h>

#include "comm.h"
#include "load.h"
#include "matrix.h"
#include "rpi.h"

#define PIPE_WEIGHTS_BASE ((uint8_t *)0x04000000)
#define PIPE_NUM_LAYERS NUM_LAYERS
#define PIPE_TX_PIN 20
#define PIPE_RX_PIN 21

void notmain(void) {
    uint8_t *layer_ptr = PIPE_WEIGHTS_BASE;

    kmalloc_init();
    void enable_dcache(void);
    enable_dcache();
    sw_uart_t uart = comm_init(PIPE_TX_PIN, PIPE_RX_PIN);

    SmolLMLayerShard layers = load_layer_shard_at(layer_ptr, PIPE_NUM_LAYERS);

    while (true) {
        Buffer *frame = comm_recv(&uart);
        Packet pkt = maybe_deserialize_packet(frame);
        free_buf(frame);
        if (pkt.matrix.rows == 0 && pkt.matrix.cols == 0) {
            free_mat(pkt.matrix);
            continue;
        }

        bool is_prefill = pkt.matrix.rows > 1;

        Matrix out;
        if (is_prefill) {
            out = block_prefill_fwd(layers.blocks[0], pkt.matrix);
            for (size_t l = 1; l < layers.num_layers; l++) {
                Matrix next = block_prefill_fwd(layers.blocks[l], out);
                free_mat(out);
                out = next;
            }
            free_mat(pkt.matrix);
        } else {
            float scratch_data[HIDDEN_SIZE];
            Buffer scratch_buf;
            Matrix scratch =
                stack_mat(&scratch_buf, scratch_data, 1, HIDDEN_SIZE);
            Matrix x = pkt.matrix;
            for (size_t l = 0; l < layers.num_layers; l++) {
                block_decode_fwd_into(layers.blocks[l], x, pkt.token_pos,
                                      scratch);
                copy(scratch, x);
            }
            out = x;
        }

        Buffer *buf = serialize_packet((Packet){
            .matrix = out,
            .token_pos = pkt.token_pos,
        });
        assert(comm_send(&uart, buf));

        free_buf(buf);
        free_mat(out);
    }
}

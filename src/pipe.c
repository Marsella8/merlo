#include <stdbool.h>

#include "comm.h"
#include "load.h"
#include "matrix.h"
#include "rpi.h"
#include "utils.h"

#define PIPE_WEIGHTS_BASE ((uint8_t *)0x04000000)
#define PIPE_NUM_LAYERS NUM_LAYERS
#define PIPE_TX_PIN 20
#define PIPE_RX_PIN 21

void notmain(void) {
    uint8_t *layer_ptr = PIPE_WEIGHTS_BASE;

    trace_tail("start");
    kmalloc_init();
    sw_uart_t uart = comm_init(PIPE_TX_PIN, PIPE_RX_PIN);

    SmolLMLayerShard layers = load_layer_shard_at(layer_ptr, PIPE_NUM_LAYERS);
    trace_tail("ready");

    while (true) {
        Buffer *frame = comm_recv(&uart);
        char *str = maybe_deserialize_string(frame);
        if (str != NULL) {
            Buffer *buf = serialize_string(str);
            assert(comm_send(&uart, buf));
            free_buf(buf);
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

        bool is_prefill = pkt.matrix.rows > 1;
        trace_tail("run %s: rows=%u cols=%u token_pos=%u",
                   is_prefill ? "prefill" : "decode",
                   (unsigned)pkt.matrix.rows,
                   (unsigned)pkt.matrix.cols,
                   (unsigned)pkt.token_pos);

        Matrix out;
        if (is_prefill) {
            out = block_prefill_fwd(layers.blocks[0], pkt.matrix);
            trace_tail("prefill layer %u/%u",
                       1u, (unsigned)layers.num_layers);
            for (size_t l = 1; l < layers.num_layers; l++) {
                Matrix next = block_prefill_fwd(layers.blocks[l], out);
                free_mat(out);
                out = next;
                trace_tail("prefill layer %u/%u",
                           (unsigned)(l + 1), (unsigned)layers.num_layers);
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
                trace_tail("decode layer %u/%u token_pos=%u",
                           (unsigned)(l + 1), (unsigned)layers.num_layers,
                           (unsigned)pkt.token_pos);
            }
            out = x;
        }
        Buffer *buf = serialize_packet((Packet){
            .matrix = out,
            .token_pos = pkt.token_pos,
        });
        assert(comm_send(&uart, buf));
        trace_tail("done %s: token_pos=%u",
                   is_prefill ? "prefill" : "decode",
                   (unsigned)pkt.token_pos);
        free_buf(buf);
        free_mat(out);
    }
}

#include <stdbool.h>

#include "comm.h"
#include "pipe.h"
#include "utils.h"

int pipe(SmolLMLayerShard layers) {
    comm_setup();

    while (true) {
        if (!string_queue_is_empty(string_queue)) {
            char* str = string_queue_deque(&string_queue);
            Buffer* buf = serialize_string(str);
            send(buf);
            free_buf(buf);
            free(str);
            continue;
        }

        if (!packet_queue_is_empty(packet_queue)) {
            Packet pkt = packet_queue_deque(&packet_queue);
            Matrix out = pkt.matrix.rows > 1
                ? layers_prefill_fwd(layers, pkt.matrix)
                : layers_decode_fwd(layers, pkt.matrix, pkt.token_pos);
            free_mat(pkt.matrix);

            Buffer* buf = serialize_packet((Packet){
                .matrix = out,
                .token_pos = pkt.token_pos,
            });
            send(buf);
            free_buf(buf);
            free_mat(out);
        }
    }

    return 0;
}

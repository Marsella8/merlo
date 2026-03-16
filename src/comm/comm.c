#include <stdarg.h>
#include <string.h>

#include "comm.h"
#include "rpi.h"

size_t const RECV_BUF_SIZE = 16 * 1024;

StringQueue string_queue = {0};
PacketQueue packet_queue = {0};
RecvState recv_state = {0};

static void reset_recv_state() {
    recv_state.expected_len = 0;
    recv_state.len_read = 0;
    recv_state.payload_read = 0;
}

static void dispatch_buffer(Buffer* frame) {
    char* str = maybe_deserialize_string(frame);
    if (str != NULL) {
        string_queue_enqueue(&string_queue, str);
        return;
    }

    Packet pkt = maybe_deserialize_packet(frame);
    if (pkt.matrix.rows != 0 || pkt.matrix.cols != 0) {
        packet_queue_enqueue(&packet_queue, pkt);
        return;
    }

    free_mat(pkt.matrix);
}

void send(Buffer* data) {
    uint32_t len = (uint32_t)data->size;
    for (int i = 0; i < 4; i++)
        uart_put8((uint8_t)(len >> (i * 8)));
    uint8_t* p = (uint8_t*)data->data;
    for (size_t i = 0; i < data->size; i++)
        uart_put8(p[i]);
}

void setup_recv() {
    recv_state.buffer = buf(RECV_BUF_SIZE);
    reset_recv_state();
    uart_init();
}

void comm_setup() {
    // TODO
}

void recv() {
    while (uart_has_data()) {
        uint8_t byte = (uint8_t)uart_get8();

        // first 4 bytes are the length of the payload
        if (recv_state.len_read < sizeof(uint32_t)) {
            recv_state.expected_len |= ((uint32_t)byte) << (recv_state.len_read * 8);
            recv_state.len_read++;
            continue;
        }

        ((uint8_t*)recv_state.buffer->data)[recv_state.payload_read++] = byte;
        if (recv_state.payload_read == recv_state.expected_len) {
            Buffer frame = {
                .data = recv_state.buffer->data,
                .size = recv_state.expected_len,
            };
            dispatch_buffer(&frame);
            reset_recv_state();
        }
    }
}

void comm_print(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintk(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (strcmp(DEVICE_NAME, "HEAD") == 0) {
        putk(buf);
    } else {
        Buffer* b = serialize_string(buf);
        send(b);
        free_buf(b);
    }
}

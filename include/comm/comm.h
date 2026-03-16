#ifndef COMM_H
#define COMM_H

#include "codec.h"
#include "matrix.h"
#include "queue.h"
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

extern StringQueue string_queue;
extern PacketQueue packet_queue;

void send(Buffer* data);
void comm_setup();

typedef struct {
    Buffer* buffer;
    uint32_t expected_len;
    size_t len_read;
    size_t payload_read;
} RecvState;

extern RecvState recv_state;

// recv is interrupt triggered
void setup_recv();
void recv();

void comm_print(const char* fmt, ...);

#endif // COMM_H

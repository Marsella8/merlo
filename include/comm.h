#ifndef COMM_H
#define COMM_H

#include <stdarg.h>
#include <stddef.h>
#include "matrix.h"
#include "codec.h"
#include "queue.h"

extern StringQueue string_queue;
extern PacketQueue packet_queue;

void send(Buffer* data);

// recv is interrupt triggered
void register_recv();
void recv();

void comm_print(const char* fmt, ...);

#endif // COMM_H

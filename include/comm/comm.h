#ifndef COMM_H
#define COMM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "queue.h"
#include "sw-uart.h"

#define COMM_BAUD 115200

sw_uart_t comm_init(uint8_t tx, uint8_t rx);
int comm_send(sw_uart_t *uart, const Buffer *payload);
Buffer *comm_recv(sw_uart_t *uart);

extern StringQueue string_queue;
extern PacketQueue packet_queue;

#endif

#ifndef COMM_H
#define COMM_H

#include "codec.h"
#include "sw-uart.h"

#define COMM_BAUD (1000000)
#define COMM_MAX_FRAME_SIZE (64 * 1024)

sw_uart_t comm_init(uint8_t tx, uint8_t rx);
int comm_send(sw_uart_t *uart, const Buffer *payload);
Buffer *comm_recv(sw_uart_t *uart);

#endif

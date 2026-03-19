#include "comm.h"

#include "cycle-count.h"
#include "rpi.h"

#define PING   0x16
#define PONG   0x06
#define BEGIN  0x02
#define DONE   0x15

#define ACK_TIMEOUT_USEC   (110 * 1000)
#define GUARD_BITS         16
#define ALLOC_GUARD_BITS   4
#define REPLY_DELAY_BITS   4

#define COMM_MAX_FRAME_SIZE BYTE_QUEUE_CAPACITY

StringQueue string_queue = {0};
PacketQueue packet_queue = {0};

static void delay_bits(sw_uart_t *uart, uint32_t nbits) {
    uint32_t start = cycle_cnt_read();
    while (cycle_cnt_read() - start < nbits * uart->cycle_per_bit)
        ;
}

static uint8_t get8_wait(sw_uart_t *uart) {
    int r;
    do {
        r = sw_uart_get8_timeout(uart, 1000 * 1000);
    } while (r < 0);
    return (uint8_t)r;
}

sw_uart_t comm_init(uint8_t tx, uint8_t rx) {
    cycle_cnt_init();
    sw_uart_t uart = sw_uart_init(tx, rx, COMM_BAUD);
    gpio_set_pullup(rx);
    return uart;
}

int comm_send(sw_uart_t *uart, const Buffer *payload) {
    assert(uart && payload);
    assert(payload->size > 0 && payload->size <= COMM_MAX_FRAME_SIZE);

    for (;;) {
        sw_uart_put8(uart, PING);
        if (sw_uart_get8_timeout(uart, ACK_TIMEOUT_USEC) == PONG)
            break;
    }

    delay_bits(uart, GUARD_BITS);
    sw_uart_put8(uart, BEGIN);

    uint32_t len = (uint32_t)payload->size;
    for (int i = 0; i < 4; i++)
        sw_uart_put8(uart, (uint8_t)(len >> (i * 8)));

    delay_bits(uart, ALLOC_GUARD_BITS);

    const uint8_t *p = payload->data;
    for (size_t i = 0; i < payload->size; i++)
        sw_uart_put8(uart, p[i]);

    return get8_wait(uart) == DONE;
}

Buffer *comm_recv(sw_uart_t *uart) {
    assert(uart);

    uint8_t b;
    do {
        b = get8_wait(uart);
        if (b == PING) {
            delay_bits(uart, REPLY_DELAY_BITS);
            sw_uart_put8(uart, PONG);
        }
    } while (b != BEGIN);

    uint8_t b0 = get8_wait(uart);
    uint8_t b1 = get8_wait(uart);
    uint8_t b2 = get8_wait(uart);
    uint8_t b3 = get8_wait(uart);
    uint32_t len = (uint32_t)b0 | ((uint32_t)b1 << 8) |
                   ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
    assert(len > 0 && len <= COMM_MAX_FRAME_SIZE);

    Buffer *frame = buf(len);
    uint8_t *data = frame->data;
    for (size_t i = 0; i < len; i++)
        data[i] = get8_wait(uart);

    delay_bits(uart, REPLY_DELAY_BITS);
    sw_uart_put8(uart, DONE);
    return frame;
}

#include "uart.h"

static bool is_initialized = false;

void uart_setup(void) {
    is_initialized = true;
}

void uart_put8(uint8_t byte) {
    (void)byte;
}

uint8_t uart_get8(void) {
    return 0;
}

bool uart_has_data(void) {
    return false;
}

#ifndef UART_H
#define UART_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

static size_t const TX = 0;
static size_t const RX = 0;

void gpio_set_output(size_t pin);
void gpio_set_input(size_t pin);

void uart_setup(void);
void uart_put8(uint8_t byte);
uint8_t uart_get8(void);
bool uart_has_data(void);

#endif

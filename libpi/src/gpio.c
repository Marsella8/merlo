/*
 * Implement the following routines to set GPIO pins to input or 
 * output, and to read (input) and write (output) them.
 *  1. DO NOT USE loads and stores directly: only use GET32 and 
 *    PUT32 to read and write memory.  See <start.S> for thier
 *    definitions.
 *  2. DO USE the minimum number of such calls.
 * (Both of these matter for the next lab.)
 *
 * See <rpi.h> in this directory for the definitions.
 *  - we use <gpio_panic> to try to catch errors.  For lab 2
 *    it only infinite loops since we don't have <printk>
 */
#include "rpi.h"
#include "rpi-interrupts.h"
// See broadcomm documents for magic addresses and magic values.
//
// If you pass addresses as:
//  - pointers use put32/get32.
//  - integers: use PUT32/GET32.
//  semantics are the same.
enum {
    // Max gpio pin number.
    GPIO_MAX_PIN = 53,

    GPIO_BASE = 0x20200000,
    gpio_fsel0 = GPIO_BASE,
    gpio_set0  = (GPIO_BASE + 0x1C),
    gpio_clr0  = (GPIO_BASE + 0x28),
    gpio_lev0  = (GPIO_BASE + 0x34),
    gpio_eds0 = (GPIO_BASE + 0x40),
    gpio_ren0 = (GPIO_BASE + 0x4C),
    gpio_fen0 = (GPIO_BASE + 0x58),

    // <you will need other values from BCM2835!>
};

//
// Part 1 implement gpio_set_on, gpio_set_off, gpio_set_output
//

// set <pin> to be an output pin.
//
// NOTE: fsel0, fsel1, fsel2 are contiguous in memory, so you
// can (and should) use ptr calculations versus if-statements!

void gpio_set_function(unsigned pin, gpio_func_t func) {
    if(pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);
    if((func & 0b111) != func)
        // NOTE: it says "func" not "function" and uses "%x" not "%d"
        gpio_panic("illegal func=%x\n", func); 
    unsigned offset = gpio_fsel0 + (pin / 10) * (sizeof(uint32_t)); // each is every 10
    unsigned shift = 3 * (pin % 10);
    unsigned mask = ~(0b111 << shift);
    unsigned read = GET32(offset);
    unsigned write = (read & mask) | (func << shift);
    PUT32(offset, write);
}

void gpio_set_output(unsigned pin) {
    gpio_set_function(pin, 0b001);
}

// Set GPIO <pin> = on.
void gpio_set_on(unsigned pin) {
    if(pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    // Implement this. 
    unsigned offset = gpio_set0 + 4*(pin / 32);
    unsigned shift = pin % 32;
    unsigned write = 1 << shift;
    PUT32(offset, write);
}


// Set GPIO <pin> = off
void gpio_set_off(unsigned pin) {
    if(pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    // Implement this. 
    unsigned offset = gpio_clr0 + 4*(pin / 32);
    unsigned shift = pin % 32;
    unsigned write = 1 << shift;
    PUT32(offset, write);
}

// Set <pin> to <v> (v \in {0,1})
void gpio_write(unsigned pin, unsigned v) {
    if(v)
        gpio_set_on(pin);
    else
        gpio_set_off(pin);
}

//
// Part 2: implement gpio_set_input and gpio_read
//

// set <pin> = input.
void gpio_set_input(unsigned pin) {
    gpio_set_function(pin, 0b000);
}

// Return 1 if <pin> is on, 0 if not.
int gpio_read(unsigned pin) {
    if(pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    unsigned offset = gpio_lev0 + (sizeof(uint32_t)) * (pin / 32);
    unsigned shift = pin % 32;
    unsigned v = (GET32(offset) >> shift) & 1;
    return v;
}

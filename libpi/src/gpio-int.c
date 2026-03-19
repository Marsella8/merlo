// engler, cs140 put your gpio-int implementations in here.
#include "rpi.h"

// in libpi/include: has useful enums.
#include "rpi-interrupts.h"

// GPIO interrupt registers (BCM2835 p96-99)
enum {
    GPIO_BASE = 0x20200000,

    gpio_eds0  = (GPIO_BASE + 0x40),  // p96: Event Detect Status
    gpio_ren0  = (GPIO_BASE + 0x4C),  // p97: Rising Edge Detect Enable
    gpio_fen0  = (GPIO_BASE + 0x58)   // p98: Falling Edge Detect Enable
};

// Enable GPIO_INT0 interrupt in IRQ_Enable_2 (p113, p117)
// GPIO_INT0 is interrupt 49, in Enable_IRQs_2 at bit (49-32)=17
// Note: caller must do dev_barrier before this since interrupt controller
// is a different device than GPIO
static void gpio_int_enable(void) {
    PUT32(IRQ_Enable_2, 1 << (GPIO_INT0 - 32));
    dev_barrier();
}

// returns 1 if there is currently a GPIO_INT0 interrupt,
// 0 otherwise.
//
// note: we can only get interrupts for <GPIO_INT0> since the
// (the other pins are inaccessible for external devices).
int gpio_has_interrupt(void) {
    dev_barrier();
    uint32_t pending = GET32(IRQ_pending_2);
    dev_barrier();
    // GPIO_INT0 is in IRQ_pending_2 (interrupts 32-63)
    return (pending >> (GPIO_INT0 - 32)) & 1;
}

// p97 set to detect rising edge (0->1) on <pin>.
// as the broadcom doc states, it  detects by sampling based on the clock.
// it looks for "011" (low, hi, hi) to suppress noise.  i.e., its triggered only
// *after* a 1 reading has been sampled twice, so there will be delay.
// if you want lower latency, you should us async rising edge (p99)
//
// also have to enable GPIO interrupts at all in <IRQ_Enable_2>
void gpio_int_rising_edge(unsigned pin) {
    if(pin>=32)
        panic("gpio_int_rising_edge: pin %d >= 32\n", pin);

    dev_barrier();
    OR32(gpio_ren0, 1 << pin);
    dev_barrier();

    gpio_int_enable();
}

// p98: detect falling edge (1->0).  sampled using the system clock.
// similarly to rising edge detection, it suppresses noise by looking for
// "100" --- i.e., is triggered after two readings of "0" and so the
// interrupt is delayed two clock cycles.   if you want  lower latency,
// you should use async falling edge. (p99)
//
// also have to enable GPIO interrupts at all in <IRQ_Enable_2>
void gpio_int_falling_edge(unsigned pin) {
    if(pin>=32)
        panic("gpio_int_falling_edge: pin %d >= 32\n", pin);

    dev_barrier();
    OR32(gpio_fen0, 1 << pin);
    dev_barrier();

    gpio_int_enable();
}

void gpio_int_falling_edge_disable(unsigned pin) {
    if(pin>=32)
        panic("gpio_int_falling_edge_disable: pin %d >= 32\n", pin);

    dev_barrier();
    uint32_t val = GET32(gpio_fen0);
    PUT32(gpio_fen0, val & ~(1 << pin));
    dev_barrier();
}

// p96: a 1<<pin is set in EVENT_DETECT if <pin> triggered an interrupt.
// if you configure multiple events to lead to interrupts, you will have to
// read the pin to determine which caused it.
int gpio_event_detected(unsigned pin) {
    if(pin>=32)
        panic("gpio_event_detected: pin %d >= 32\n", pin);

    dev_barrier();
    uint32_t val = GET32(gpio_eds0);
    dev_barrier();

    return (val >> pin) & 1;
}

// p96: have to write a 1 to the pin to clear the event.
void gpio_event_clear(unsigned pin) {
    if(pin>=32)
        panic("gpio_event_clear: pin %d >= 32\n", pin);

    dev_barrier();
    PUT32(gpio_eds0, 1 << pin);
    dev_barrier();
}

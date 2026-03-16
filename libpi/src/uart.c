// simple mini-uart driver: implement every routine 
// with a <todo>.
//
// NOTE: 
//  - from broadcom: if you are writing to different 
//    devices you MUST use a dev_barrier().   
//  - its not always clear when X and Y are different
//    devices.
//  - pay attenton for errata!   there are some serious
//    ones here.  if you have a week free you'd learn 
//    alot figuring out what these are (esp hard given
//    the lack of printing) but you'd learn alot, and
//    definitely have new-found respect to the pioneers
//    that worked out the bcm eratta.
//
// historically a problem with writing UART code for
// this class (and for human history) is that when 
// things go wrong you can't print since doing so uses
// uart.  thus, debugging is very old school circa
// 1950s, which modern brains arne't built for out of
// the box.   you have two options:
//  1. think hard.  we recommend this.
//  2. use the included bit-banging sw uart routine
//     to print.   this makes things much easier.
//     but if you do make sure you delete it at the 
//     end, otherwise your GPIO will be in a bad state.
//
// in either case, in the next part of the lab you'll
// implement bit-banged UART yourself.
#include "rpi.h"

// change "1" to "0" if you want to comment out
// the entire block.
#if 1
//*****************************************************
// We provide a bit-banged version of UART for debugging
// your UART code.  delete when done!
//
// NOTE: if you call <emergency_printk>, it takes 
// over the UART GPIO pins (14,15). Thus, your UART 
// GPIO initialization will get destroyed.  Do not 
// forget!   

// header in <libpi/include/sw-uart.h>
#include "sw-uart.h"
static sw_uart_t sw_uart;

// a sw-uart putc implementation.
static int sw_uart_putc(int chr) {
    sw_uart_put8(&sw_uart,chr);
    return chr;
}
// static void uart_flush_rx(void) {
//     while (uart_has_data()) {
//         uart_get8_async();
//     }
// }

// L -> Line, C -> Control, S -> Status

uint32_t AUX_ENB = 0x20215004;
uint32_t AUX_MU_IO = 0x20215040;
uint32_t AUX_MU_IER = 0x20215044;
uint32_t AUX_MU_IIR = 0x20215048;
uint32_t AUX_MU_LCR = 0x2021504C;
uint32_t AUX_MU_LSR = 0x20215054;
uint32_t AUX_MU_CNTL = 0x20215060;
uint32_t AUX_MU_STAT = 0x20215064;
uint32_t AUX_MU_BAUD = 0x20215068;

static void rmw(unsigned address, uint32_t mask, uint32_t value) {
    uint32_t v = (uint32_t)GET32(address);
    v = (v & ~mask) | (value & mask);
    PUT32(address, v);
}

// call this routine to print stuff. 
//
// note the function pointer hack: after you call it 
// once can call the regular printk etc.
__attribute__((noreturn)) 
static void emergency_printk(const char *fmt, ...)  {
    // we forcibly initialize in case the 
    // GPIO got reset. this will setup 
    // gpio 14,15 for sw-uart.
    sw_uart = sw_uart_default();

    // all libpi output is via a <putc>
    // function pointer: this installs ours
    // instead of the default
    rpi_putchar_set(sw_uart_putc);

    // do print
    va_list args;
    va_start(args, fmt);
    vprintk(fmt, args);
    va_end(args);

    // at this point UART is all messed up b/c we took it over
    // so just reboot.   we've set the putchar so this will work
    clean_reboot();
}

#undef todo
#define todo(msg) do {                          \
    emergency_printk("%s:%d:%s\nDONE!!!\n",     \
            __FUNCTION__,__LINE__,msg);         \
} while(0)

// END of the bit bang code.
#endif


//*****************************************************
// the rest you should implement.

// called first to setup uart to 8n1 115200  baud,
// no interrupts.
//  - you will need memory barriers, use <dev_barrier()>
//
//  later: should add an init that takes a baud rate.
void uart_init(void) {
    dev_barrier();
    
    gpio_set_function(14, GPIO_FUNC_ALT5);
    gpio_set_function(15, GPIO_FUNC_ALT5);
    
    dev_barrier();
    
    rmw(AUX_ENB, 0b1, 0b1); // enable Uart
    
    dev_barrier();
    
    PUT32(AUX_MU_CNTL, 0b0); //disable TX, RX
    
    PUT32(AUX_MU_IIR, 0b110); // flush

    PUT32(AUX_MU_IER, 0b0); // disable interrupts

    PUT32(AUX_MU_LCR, 0b11);

    size_t baud_rate = 115200;
    size_t system_clock = 250000000;
    size_t baudrate_counter = system_clock / (8 * baud_rate) - 1;
    PUT32(AUX_MU_BAUD, baudrate_counter);

    PUT32(AUX_MU_CNTL, 0b11);

    dev_barrier();
}

// disable the uart: make sure all bytes have been
// 
void uart_disable(void) {
    uart_flush_tx();
    rmw(AUX_ENB, 0b1, 0b0);
    
}

// returns one byte from the RX (input) hardware
// FIFO.  if FIFO is empty, blocks until there is 
// at least one byte.
int uart_get8(void) {
    dev_barrier();
    while (!uart_has_data()) {
        rpi_wait(); //spin
    }
    uint32_t v = GET32(AUX_MU_IO) & 0xFF;
    dev_barrier();
    return (int)v;
}

// returns 1 if the hardware TX (output) FIFO has room
// for at least one byte.  returns 0 otherwise.
int uart_can_put8(void) {
    // TODO: implement this!
    uint32_t v = (GET32(AUX_MU_STAT) >> 1) & 0b1;
    return (int)v;
}

// put one byte on the TX FIFO, if necessary, waits
// until the FIFO has space.
int uart_put8(uint8_t c) {
    while (!uart_can_put8()) {
        rpi_wait();
    }
    rmw(AUX_MU_IO, 0xFF, (uint32_t)c);
    return 0; // ???
}

// returns:
//  - 1 if at least one byte on the hardware RX FIFO.
//  - 0 otherwise
int uart_has_data(void) {
    uint32_t v = (GET32(AUX_MU_STAT) >> 0) & 0b1;
    return (int)v;
}

// returns:
//  -1 if no data on the RX FIFO.
//  otherwise reads a byte and returns it.
int uart_get8_async(void) { 
    if(!uart_has_data())
        return -1;
    return uart_get8();
}

// returns:
//  - 1 if TX FIFO empty AND idle.
//  - 0 if not empty.
int uart_tx_is_empty(void) {
    unsigned v = (GET32(AUX_MU_STAT) >> 9) & 0b1;
    return (int)v;
}

// return only when the TX FIFO is empty AND the
// TX transmitter is idle.  
//
// used when rebooting or turning off the UART to
// make sure that any output has been completely 
// transmitted.  otherwise can get truncated 
// if reboot happens before all bytes have been
// received.
void uart_flush_tx(void) {
    while(!uart_tx_is_empty())
        rpi_wait();
}

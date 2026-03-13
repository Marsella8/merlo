#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "comm.h"
#include "uart.h"

#ifndef DEVICE_NAME
#define DEVICE_NAME "HEAD"
#endif

StringQueue string_queue = {0};
PacketQueue packet_queue = {0};

void send(Buffer* data) {
    uint32_t len = (uint32_t)data->size;
    for (int i = 0; i < 4; i++)
        uart_put8((uint8_t)(len >> (i * 8)));
    uint8_t* p = (uint8_t*)data->data;
    for (size_t i = 0; i < data->size; i++)
        uart_put8(p[i]);
}

void recv() {
}

void comm_print(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);
    if (strcmp(DEVICE_NAME, "HEAD") == 0) {
        printf("%s", buf);
        fflush(stdout);
    } else {
        Buffer* b = serialize_string(buf);
        send(b);
        free_buf(b);
    }
}

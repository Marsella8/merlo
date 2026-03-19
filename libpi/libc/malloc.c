#include "rpi.h"

void *malloc(size_t nbytes) {
    return kmalloc_notzero((unsigned)nbytes);
}

void free(void *ptr) {
    (void)ptr;
}

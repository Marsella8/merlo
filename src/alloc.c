#include "alloc.h"
#include "rpi.h"

void* malloc(size_t size) {
    return kmalloc((unsigned)size);
}

// libpi uses a bump allocator!!!! so tjis is a no-op.
void free(void* ptr) {
    (void)ptr;
}

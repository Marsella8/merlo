#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

static inline void panic(const char* message) {
    fprintf(stderr, "%s, %d: %s\n", __FILE__, __LINE__, message);
    exit(1);
}

static inline void not_implemented() {
    fprintf(stderr, "%s, %d: NOT IMPLEMENTED\n", __FILE__, __LINE__);
    exit(1);
}

char* substr(const char* str, size_t start, size_t end); // make sure to free!
#endif // UTILS_H
#include "utils.h"

void panic(const char* message) {
    fprintf(stderr, "%s, %d: %s\n", __FILE__, __LINE__, message);
    exit(1);
}

void not_implemented() {
    fprintf(stderr, "%s, %d: NOT IMPLEMENTED\n", __FILE__, __LINE__);
    exit(1);
}
#include <math.h>
#include <string.h>

#include "rpi.h"
#include "utils.h"

int* __errno(void) {
    static int err;
    return &err;
}

void not_implemented_impl(const char* file, int line) {
    panic("%s:%d: NOT IMPLEMENTED", (char*)file, line);
}

void trace_impl(const char* label, const char* file, int line, const char* fmt, ...) {
    va_list args;
    printk("%s:%s:%d: ", (char*)label, (char*)file, line);
    va_start(args, fmt);
    vprintk(fmt, args);
    va_end(args);
    printk("\n");
}

void warning_impl(const char* file, int line, const char* message) {
    printk("%s:%d: WARNING: %s\n", (char*)file, line, (char*)message);
}

void assume_no_nans_impl(Matrix m, const char* file, int line) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            if (isnan(*at(m, i, j))) {
                panic("%s:%d: NaN detected", (char*)file, line);
            }
        }
    }
}

size_t ceil_div(size_t x, size_t y) {
    return (x + y - 1) / y;
}

char* substr(const char* str, size_t start, size_t end) {
    size_t len = end - start;
    char* result = malloc(len + 1);
    memcpy(result, str + start, len);
    result[len] = '\0';
    return result;
}

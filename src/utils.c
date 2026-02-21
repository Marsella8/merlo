#include "utils.h"
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

void panic_impl(const char* file, int line, const char* message) {
    fprintf(stderr, "%s:%d: %s\n", file, line, message);
    exit(1);
}

void not_implemented_impl(const char* file, int line) {
    fprintf(stderr, "%s:%d: NOT IMPLEMENTED\n", file, line);
    exit(1);
}

void warning_impl(const char* file, int line, const char* message) {
    fprintf(stderr, "%s:%d: WARNING: %s\n", file, line, message);
}

void assume_no_nans_impl(Matrix m, const char* file, int line) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            if (isnan(*at(m, i, j))) {
                char message[128];
                snprintf(message, sizeof(message), "NaN detected at (%zu, %zu)", i, j);
                panic_impl(file, line, message);
            }
        }
    }
}

char* substr(const char* str, size_t start, size_t end) {
    size_t len = end - start;
    char* result = malloc(len + 1);
    assert(result != NULL);
    memcpy(result, str + start, len);
    result[len] = '\0';
    return result;
}

#include "utils.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>

char* substr(const char* str, size_t start, size_t end) {
    size_t len = end - start;
    char* result = malloc(len + 1);
    assert(result != NULL);
    memcpy(result, str + start, len);
    result[len] = '\0';
    return result;
}
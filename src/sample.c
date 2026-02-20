#include "sample.h"
#include "matrix.h"
#include "model.h"
#include "utils.h"
#include <assert.h>

size_t argmax(Matrix logits) {
#ifdef SAFETY
    assume_shape(logits, 1, VOCAB_SIZE);
#endif
    size_t idx = 0;
    for (size_t i = 1; i < VOCAB_SIZE; i++) {
        if (*at(logits, 0, i) > *at(logits, 0, idx)) {
            idx = i;
        }
    }
    return idx;
}

size_t sample(Matrix logits) {
    panic("Not implemented");
    return 0;
}
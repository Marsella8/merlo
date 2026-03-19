#include <stdint.h>

#include "nn.h"
#include "sample.h"

static uint32_t rand_state = 1;

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

float rand_float() {
    rand_state = rand_state * 1103515245u + 12345u;
    return (float)(rand_state & 0x7fffffffu) / 2147483648.0f;
}

size_t sample(Matrix logits, float temperature) {
#ifdef SAFETY
    assume_shape(logits, 1, VOCAB_SIZE);
    assert(temperature > 0.0f);
#endif
    float probs_data[VOCAB_SIZE];
    Buffer probs_buf;
    Matrix probs = stack_mat(&probs_buf, probs_data, 1, VOCAB_SIZE);
    scale_into(logits, 1.0f / temperature, probs);
    softmax_into(probs, probs);

    float r = rand_float();
    float cum = 0.0f;
    for (size_t i = 0; i < (size_t)VOCAB_SIZE; i++) {
        cum += *at(probs, 0, i);
        if (r <= cum) {
            return i;
        }
    }

    return (size_t)VOCAB_SIZE - 1;
}

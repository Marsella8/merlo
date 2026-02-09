#include "tokenizer.h"
#include "utils.h"
#include <string.h>
#include <stdlib.h>

Vocab load_vocab() {
    not_implemented();
    return (Vocab){0};
}

void free_vocab(Vocab* v) {
    if (v->tokens) {
        for (size_t i = 0; i < v->size; i++) {
            if (v->tokens[i]) free(v->tokens[i]);
        }
        free(v->tokens);
    }
    v->tokens = NULL;
    v->size = 0;
}

Matrix tokenize(const char* string, const Vocab* v) {
    not_implemented();
    return (Matrix){0};
}

char* detokenize(Matrix tokens, const Vocab* v) {
#ifdef SAFETY
    if (tokens.rows != 1) {
        panic("Matrix must have 1 row in detokenize()\n");
    }
#endif
    not_implemented();
    return NULL;
}

char* detokenize_token(size_t token_id, const Vocab* v) {
    not_implemented();
    return NULL;
}


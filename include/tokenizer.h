#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "matrix.h"
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    char **tokens;
    size_t size;
} Vocab;

Vocab load_vocab();
void free_vocab(Vocab* v);
Matrix tokenize(const char* string, const Vocab* v);
char* detokenize(Matrix tokens, const Vocab* v);
char* detokenize_token(size_t token_id, const Vocab* v);

#endif // TOKENIZER_H

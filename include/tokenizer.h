#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "matrix.h"
#include <stddef.h>
#include <stdbool.h>

static const char* const VOCAB_PATH = "model/vocab.txt";

typedef struct {
    char **tokens;
    size_t size;
} Vocab;

Vocab load_vocab();
void free_vocab(Vocab v);

int tokenize_single_token(char* string, Vocab v);
Matrix tokenize(const char* string, Vocab v);

char* detokenize(Matrix tokens, Vocab v);

#endif // TOKENIZER_H

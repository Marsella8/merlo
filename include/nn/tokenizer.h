#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "matrix.h"

const uint8_t *lookup_token(size_t token_id);
size_t lookup_token_len(size_t token_id);
void print_token(size_t token_id);
void tokenizer_set_storage(const uint8_t *base, size_t vocab_size);
void tokenizer_reset_storage(void);
int tokenize_single_token(const char *string);
Matrix tokenize(const char *string);
char *detokenize(Matrix tokens);

#endif // TOKENIZER_H

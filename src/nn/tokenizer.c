#include <stdint.h>
#include <string.h>

#include "model.h"
#include "rpi.h"
#include "tokenizer.h"
#include "utils.h"

#define TOKENIZER_OFFSET_BASE ((const uint8_t *)0x04000000)
#define TOKEN_SIZE_BYTES 256

static const int NOT_A_TOKEN = -1;
static const uint8_t *tokenizer_base = TOKENIZER_OFFSET_BASE;
static size_t tokenizer_vocab_size = VOCAB_SIZE;

void tokenizer_set_storage(const uint8_t *base, size_t vocab_size) {
#ifdef SAFETY
    assert(base != NULL);
    assert(vocab_size > 0);
#endif
    tokenizer_base = base;
    tokenizer_vocab_size = vocab_size;
}

void tokenizer_reset_storage(void) {
    tokenizer_base = TOKENIZER_OFFSET_BASE;
    tokenizer_vocab_size = VOCAB_SIZE;
}

const uint8_t *lookup_token(size_t token_id) {
    return tokenizer_base + token_id * TOKEN_SIZE_BYTES;
}

size_t lookup_token_len(size_t token_id) {
    const uint8_t *token = lookup_token(token_id);
    for (size_t i = TOKEN_SIZE_BYTES; i > 0; i--) {
        if (token[i - 1] != 0)
            return i;
    }
    return 1;
}

void print_token(size_t token_id) {
    const uint8_t *token = lookup_token(token_id);
    size_t len = lookup_token_len(token_id);
    for (size_t i = 0; i < len; i++)
        uart_put8(token[i]);
}

static bool token_matches_string(size_t token_id, const char *string, size_t string_len) {
    return lookup_token_len(token_id) == string_len &&
           memcmp(lookup_token(token_id), string, string_len) == 0;
}

int tokenize_single_token(const char *string) {
    if (string == NULL)
        panic("tokenize_single_token(): string must not be NULL\n");
    size_t string_len = strlen(string);
    for (size_t i = 0; i < tokenizer_vocab_size; i++) {
        if (token_matches_string(i, string, string_len))
            return i;
    }
    return NOT_A_TOKEN;
}

Matrix tokenize(const char *string) {
    size_t len = strlen(string);
    size_t num_bytes = len * sizeof(size_t);
    size_t *rle = malloc(num_bytes);
    for (size_t i = 0; i < len; i++)
        rle[i] = 1u;

    while (1) {
        size_t min_rank = SIZE_MAX;
        size_t pos = SIZE_MAX;
        size_t offset = 0;
        for (size_t i = 0; i < len - 1; i++) {
            size_t span = rle[i] + rle[i + 1];
            char *merged = substr(string, offset, offset + span);
            int token_id = tokenize_single_token(merged);
            free(merged);
            if (token_id != NOT_A_TOKEN && (size_t)token_id < min_rank) {
                min_rank = token_id;
                pos = i;
            }
            offset += rle[i];
        }
        if (min_rank == SIZE_MAX)
            break;
        rle[pos] += rle[pos + 1];
        len--;
        memmove(rle + pos + 1, rle + pos + 2, (len - pos - 1) * sizeof(size_t));
    }

    Buffer *b = buf(len * sizeof(float));
    float *data = (float *)b->data;
    size_t offset = 0;
    for (size_t i = 0; i < len; i++) {
        char *token_str = substr(string, offset, offset + rle[i]);
        int token_id = tokenize_single_token(token_str);
        free(token_str);
        if (token_id == NOT_A_TOKEN)
            panic("tokenize(): unknown token span\n");
        data[i] = (float)token_id;
        offset += rle[i];
    }
    free(rle);
    return mat(b, 1, len);
}

static size_t token_id_at(Matrix tokens, size_t i) {
    float val = *at(tokens, 0, i);
#ifdef SAFETY
    assert(val >= 0.0f);
    size_t id = (size_t)val;
    assert((float)id == val);
    assert(id < tokenizer_vocab_size);
#else
    size_t id = (size_t)val;
    if (id >= tokenizer_vocab_size)
        panic("detokenize(): token id out of bounds\n");
#endif
    return id;
}

char *detokenize(Matrix tokens) {
#ifdef SAFETY
    if (tokens.rows != 1)
        panic("Matrix must have 1 row in detokenize()\n");
#endif
    size_t total_bytes = 0;
    for (size_t i = 0; i < tokens.cols; i++)
        total_bytes += lookup_token_len(token_id_at(tokens, i));

    char *result = malloc(total_bytes + 1);
    size_t offset = 0;
    for (size_t i = 0; i < tokens.cols; i++) {
        size_t id = token_id_at(tokens, i);
        size_t token_len = lookup_token_len(id);
        memcpy(result + offset, lookup_token(id), token_len);
        offset += token_len;
    }
    result[offset] = '\0';
    return result;
}

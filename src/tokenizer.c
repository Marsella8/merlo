#include "tokenizer.h"
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>

Vocab load_vocab() {
    FILE *fp = fopen(VOCAB_PATH, "rb");
    assert(fp != NULL);

    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    rewind(fp);

    assert(file_size % 256 == 0);
    size_t num_tokens = file_size / 256;
    char** tokens = malloc(sizeof(char*) * num_tokens);
    for (size_t i = 0; i < num_tokens; i++) {
        tokens[i] = malloc(256);
        fread(tokens[i], 1, 256, fp);
    }
    fclose(fp);
    return (Vocab){.tokens = tokens, .size = num_tokens};
}

void free_vocab(Vocab v) {
    if (v.tokens) {
        for (size_t i = 0; i < v.size; i++) {
            if (v.tokens[i]) free(v.tokens[i]);
        }
        free(v.tokens);
    }
}

// no optionals, fuck! don't like this code
const int NOT_A_TOKEN = -1;
int tokenize_single_token(char* string, const Vocab* v) {
    for (size_t i = 0; i<v->size; i++) {
        if (strcmp(string, v->tokens[i])==0)
            return i;
    }
    return NOT_A_TOKEN;
}


Matrix tokenize(const char* string, const Vocab* v) {
    size_t len = strlen(string);
    size_t* rle = malloc(sizeof(size_t) * len); // run length encoding
    for (size_t i=0;i<len;i++) {rle[i]=1;}
    while (1) {

        // find merge pair with lowest rank (i thought you could do BPE greedily, but I was wrong (chud me))
        size_t min_rank = SIZE_MAX;
        size_t pos = -1;
        size_t offset = 0;
        for (size_t i = 0; i<len-1;i++) {
            size_t len = rle[i] + rle[i+1];
            char* merged_token = substr(string, offset, offset + len);
            int token_id = tokenize_single_token(merged_token, v);
            free(merged_token);
            if (token_id != NOT_A_TOKEN) {
                if ((size_t)token_id < min_rank) {
                    min_rank = token_id;
                    pos = i;
                }
            }
            offset += rle[i];
        }
        if (min_rank == SIZE_MAX)
            break; // no merge pairs found, we are done 
        rle[pos] = rle[pos] + rle[pos+1];
        len--;
        memmove(rle + pos + 1, rle + pos + 2, (len - pos - 1) * sizeof(size_t));
    }
    Buffer* b = buf(len * sizeof(size_t));
    float* data = (float*)b->data;
    size_t current_offset = 0;
    for (size_t i = 0; i < len; i++) {
        char* token_str = substr(string, current_offset, current_offset + rle[i]);
        data[i] = (float)tokenize_single_token(token_str, v); // should really be an int
        current_offset += rle[i];
        free(token_str);
    }
    free(rle);
    return mat(b, 1, len);
}

char* detokenize(Matrix tokens, const Vocab* v) {
#ifdef SAFETY
    if (tokens.rows != 1) {
        panic("Matrix must have 1 row in detokenize()\n");
    }
#endif
    char* result = malloc(tokens.cols * 256);
    result[0] = '\0';
    for (size_t i = 0; i < tokens.cols; i++) {
        size_t token_id = (size_t)*at(tokens, 0, i);
        char* token_str = v->tokens[token_id];
        strcat(result, token_str);
    }
    return result;
}

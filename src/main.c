#include <stdio.h>
#include <string.h>
#include "model.h"
#include "tokenizer.h"
#include "prefill.h"
#include "engine.h"
#include "sample.h"

int main() {
    const char* prompt = "Hello my name is ";
    SmolLM2 model = load_model();
    Vocab vocab = load_vocab();
    Matrix tokens = tokenize(prompt, &vocab);
    KVCache cache = prefill(model, tokens);
    printf(prompt);
    for (int pos = tokens.cols; pos < MAX_SEQ_LEN; pos++) {
        size_t last_token_id = (size_t)*at(tokens, 0, tokens.cols - 1);
        size_t token_id = fwd(model, &cache, last_token_id, pos);
        
        char* str = detokenize_token(token_id, &vocab);
        printf("%s", str);
        fflush(stdout);
    }

    free_model(&model);
    free_vocab(&vocab);
    free_buf(tokens.buffer);
    // Note: KVCache buffers

    return 0;
}
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
    printf("%s", prompt);
    for (int pos = tokens.cols; pos < MAX_SEQ_LEN; pos++) {
        size_t last_token_id = (size_t)*at(tokens, 0, tokens.cols - 1);
        Matrix logits = fwd(model, model.cache, last_token_id, pos);
        size_t token = argmax(logits);
        char* str = vocab.tokens[token];
        printf("%s", str);
        fflush(stdout);
        free_mat(logits);
    }

    free_model(model);
    free_vocab(vocab);
    free_mat(tokens);

    return 0;
}
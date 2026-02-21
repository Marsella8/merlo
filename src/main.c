#include <stdio.h>
#include <string.h>
#include "model.h"
#include "tokenizer.h"
#include "sample.h"

int main() {
    const char* prompt = "Hello my name is ";
    SmolLM2 model = load_model();
    Vocab vocab = load_vocab();
    Matrix tokens = tokenize(prompt, vocab);
    prefill(model, tokens, 0);
    printf("%s", prompt);
    size_t last_token_id = (size_t)*at(tokens, 0, tokens.cols - 1);
    for (size_t pos = tokens.cols; pos < MAX_SEQ_LEN; pos++) {
        Matrix logits = fwd(model, last_token_id, pos);
        size_t token = argmax(logits);
        char* str = vocab.tokens[token];
        fflush(stdout);
        free_mat(logits);
        last_token_id = token;
    }

    free_model(model);
    free_vocab(vocab);
    free_mat(tokens);

    return 0;
}

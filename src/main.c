#include <stdio.h>
#include <string.h>
#include "model.h"
#include "tokenizer.h"
#include "sample.h"

int main() {
    const char* prompt = "The color of the sky is ";
    size_t max_tokens = 50;
    SmolLM2 model = load_main_model();
    Vocab vocab = load_vocab();
    Matrix tokens = tokenize(prompt, vocab);
    prefill(model, tokens);
    printf("%s", prompt);
    size_t last_token_id = (size_t)*at(tokens, 0, tokens.cols - 1);
    for (size_t pos = tokens.cols; pos < max_tokens; pos++) {
        Matrix logits = fwd(model, last_token_id, pos);
        size_t token = sample(logits, 0.9f);
        char* str = vocab.tokens[token];
        printf("%s", str);
        fflush(stdout);
        last_token_id = token;
        free_mat(logits);
    }

    free_model(model);
    free_vocab(vocab);
    free_mat(tokens);

    return 0;
}

#include "model.h"
#include "rpi.h"
#include "sample.h"
#include "tokenizer.h"

void notmain(void) {
    kmalloc_init();
    const char* prompt = "Dogs are the most";
    size_t max_tokens = 100;
    SmolLM2 model = load_main_model();
    Vocab vocab = load_vocab();
    Matrix tokens = tokenize(prompt, vocab);
    prefill(model, tokens);
    putk(prompt);
    size_t last_token_id = (size_t)*at(tokens, 0, tokens.cols - 1);
    for (size_t pos = tokens.cols; pos < max_tokens; pos++) {
        Matrix logits = fwd(model, last_token_id, pos);
        size_t token = sample(logits, 0.9f);
        char* str = vocab.tokens[token];
        putk(str);
        last_token_id = token;
        free_mat(logits);
    }

    free_model(model);
    free_vocab(vocab);
    free_mat(tokens);
}

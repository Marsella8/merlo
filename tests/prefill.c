#include <stdio.h>
#include <assert.h>
#include "model.h"
#include "matrix.h"
#include "tokenizer.h"
#include "prefill.h"
#include "nn.h"

void test_prefill() {
    printf("  test_prefill\n");
    SmolLM2 model = load_model();
    Vocab vocab = load_vocab();

    const char* prompt = "Hello my name is";
    Matrix tokens = tokenize(prompt, &vocab);

    prefill(&model, tokens);

    free_mat(tokens);
    free_model(model);
    free_vocab(vocab);
}

int main() {
    printf("prefill tests:\n");
    test_prefill();
    return 0;
}

#include <stdint.h>
#include <string.h>

#include "rpi.h"
#include "tokenizer.h"

#define TOKEN_SIZE_BYTES 256

typedef struct {
    char **tokens;
    size_t size;
} Vocab;

static char *tokens_arr[] = {"a", "b", "aa", "bb", "ba", "ab", "aaa", "aab", "baa", "bab", "bba", "bbb"};
static Vocab vocab = {
    .tokens = tokens_arr,
    .size = 12,
};

static uint8_t *packed_vocab = NULL;

static uint8_t *pack_vocab(Vocab test_vocab) {
    uint8_t *packed = malloc(test_vocab.size * TOKEN_SIZE_BYTES);
    memset(packed, 0, test_vocab.size * TOKEN_SIZE_BYTES);
    for (size_t i = 0; i < test_vocab.size; i++) {
        size_t token_len = strlen(test_vocab.tokens[i]);
        memcpy(packed + i * TOKEN_SIZE_BYTES, test_vocab.tokens[i], token_len);
    }
    return packed;
}

static void install_test_vocab(void) {
    packed_vocab = pack_vocab(vocab);
    tokenizer_set_storage(packed_vocab, vocab.size);
}


static void test_tokenize_single_token(void) {
    printk("  test_tokenize_single_token\n");

    {
        const char *input = "baa";
        int correct = 8;
        int actual = tokenize_single_token(input);
        assert(actual == correct);
    }

    {
        const char *input = "b";
        int correct = 1;
        int actual = tokenize_single_token(input);
        assert(actual == correct);
    }

    {
        const char *input = "c";
        int correct = -1;
        int actual = tokenize_single_token(input);
        assert(actual == correct);
    }
}

static void test_tokenize(void) {
    printk("  test_tokenize\n");

    {
        const char *input = "aaabbba";
        float correct_data[1][3] = {{6.0f, 3.0f, 4.0f}};
        Matrix correct = mat_from_array(1, 3, correct_data);
        Matrix actual = tokenize(input);

        assert(eq(actual, correct));

        free_mat(actual);
        free_mat(correct);
    }

    {
        const char *input = "ababa";
        float correct_data[1][3] = {{0.0f, 4.0f, 4.0f}};
        Matrix correct = mat_from_array(1, 3, correct_data);
        Matrix actual = tokenize(input);

        assert(eq(actual, correct));

        free_mat(actual);
        free_mat(correct);
    }

    {
        const char *input = "aaaaa";
        float correct_data[1][2] = {{2.0f, 6.0f}};
        Matrix correct = mat_from_array(1, 2, correct_data);
        Matrix actual = tokenize(input);

        assert(eq(actual, correct));

        free_mat(actual);
        free_mat(correct);
    }
}

static void test_detokenize(void) {
    printk("  test_detokenize\n");

    {
        float tokens_data[1][3] = {{8.0f, 3.0f, 4.0f}}; // "baa", "bb", "ba" -> "baabbba"
        Matrix tokens = mat_from_array(1, 3, tokens_data);
        char *actual = detokenize(tokens);
        assert(strcmp(actual, "baabbba") == 0);
        free(actual);
        free_mat(tokens);
    }
}

void notmain(void) {
    kmalloc_init();
    install_test_vocab();
    printk("tokenizer tests:\n");
    test_tokenize_single_token();
    test_tokenize();
    test_detokenize();
    printk("tests/single/nn/tokenizer: pass\n");
    clean_reboot();
}

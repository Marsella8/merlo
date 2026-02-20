#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include "tokenizer.h"
#include "matrix.h"

char* tokens_arr[] = {"a", "b", "aa", "bb", "ba", "ab", "aaa", "aab", "baa", "bab", "bba", "bbb"};
Vocab vocab = {
    .tokens = tokens_arr,
    .size = 12
};

void test_tokenize_single_token() {
    printf("  test_tokenize_single_token\n");
    
    {
        const char* input = "baa";
        int correct = 6; 
        int output = tokenize_single_token((char*)input, &vocab);
        assert(output == correct);
    }

    {
        const char* input = "b";
        int correct = 1;
        int output = tokenize_single_token((char*)input, &vocab);
        assert(output == correct);
    }

    {
        const char* input = "c";
        int correct = -1;
        int output = tokenize_single_token((char*)input, &vocab);
        assert(output == correct);
    }
}

void test_tokenize() {
    printf("  test_tokenize\n");

    {
        const char* input = "aaabbba";
        float correct_data[1][3] = {{6.0f, 3.0f, 4.0f}};
        Matrix correct = mat_from_array(1, 3, correct_data);
        Matrix output = tokenize(input, &vocab);

        assert(eq(output, correct));
        
        free_mat(output);
        free_mat(correct);
    }

    {
        const char* input = "ababa";
        float correct_data[1][3] = {{0.0f, 4.0f, 4.0f}};
        Matrix correct = mat_from_array(1, 3, correct_data);
        Matrix output = tokenize(input, &vocab);

        assert(eq(output, correct));

        free_mat(output);
        free_mat(correct);
    }

    {
        const char* input = "aaaaa";
        float correct_data[1][2] = {{2.0f, 6.0f}};
        Matrix correct = mat_from_array(1, 2, correct_data);
        Matrix output = tokenize(input, &vocab);

        assert(eq(output, correct));

        free_mat(output);
        free_mat(correct);
    }
}

void test_load_vocab() {
    printf("  test_load_vocab\n");
    Vocab v = load_vocab();
    
    {
        size_t input = 0;
        const char* correct = "<|endoftext|>";
        const char* output = v.tokens[input];
        assert(strcmp(output, correct) == 0);
    }
    {
        size_t input = 1000;
        const char* correct = "()";
        const char* output = v.tokens[input];
        assert(strcmp(output, correct) == 0);
    }
    {
        size_t input = 49151;
        const char* correct = "ectable";
        const char* output = v.tokens[input];
        assert(strcmp(output, correct) == 0);
    }
    
    free_vocab(v);
}

void test_tokenize_full() {
    printf("  test_tokenize_full\n");
    Vocab v = load_vocab();
    
    {
        const char* input = "C and its consequences have been disastrous for the human race";
        float correct_data[1][11] = {{51.0f, 284.0f, 624.0f, 4748.0f, 457.0f, 719.0f, 24548.0f, 327.0f, 260.0f, 1205.0f, 4677.0f}};

        Matrix correct = mat_from_array(1, 11, correct_data);
        
        Matrix output = tokenize(input, &v);
        assert(eq(output, correct));
        free_mat(output);
        free_mat(correct);
    }
    
    free_vocab(v);
}

void test_detokenize() {
    printf("  test_detokenize\n");

    {
        float tokens_data[1][3] = {{8.0f, 3.0f, 4.0f}}; // "baa", "bb", "ba" -> "baabbba"
        Matrix tokens = mat_from_array(1, 3, tokens_data);
        char* output = detokenize(tokens, &vocab);
        assert(strcmp(output, "baabbba") == 0);
        free(output);
        free_mat(tokens);
    }
}

void test_detokenize_full() {
    printf("  test_detokenize_full\n");
    Vocab v = load_vocab();
    
    {
        const char* input = "C and its consequences have been disastrous for the human race";
        // kinda not fully sound because tokenization is not bijective
        Matrix tokens = tokenize(input, &v);
        char* output = detokenize(tokens, &v);
        assert(strcmp(output, input) == 0);
        free(output);
        free_mat(tokens);
    }
    
    free_vocab(v);
}

int main() {
    printf("tokenizer tests:\n");
    test_tokenize();
    test_detokenize();
    test_load_vocab();
    test_tokenize_full();
    test_detokenize_full();
    return 0;
}

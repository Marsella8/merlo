#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "tokenizer.h"
#include "matrix.h"

char* tokens_arr[] = {"a", "b", "aa", "bb", "aaa", "aab", "baa", "bab", "bba", "bbb"};
Vocab vocab = {
    .tokens = tokens_arr,
    .size = 10
};


void test_tokenize() {
    printf("test_tokenize\n");
    const char* string = "aaabbba";
    Matrix tokens = tokenize(string, &vocab);
    
    float correct_data[1][3] = {{4.0f, 9.0f, 0.0f}};
    Matrix correct = mat_from_array(1, 3, correct_data);
    
    assert(eq(tokens, correct));
    
    free_buf(tokens.buffer);
    free_buf(correct.buffer);
}

int main() {
    test_tokenize();
    printf("Done!\n");
    return 0;
}

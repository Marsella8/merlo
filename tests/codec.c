#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codec.h"

void test_string_roundtrip() {
    printf("  test_string_roundtrip\n");
    const char* input = "Church";
    const char* correct = input;
    Buffer* buf = serialize_string((char*)input);
    char* output = maybe_deserialize_string(buf);
    assert(output != NULL);
    assert(strcmp(output, correct) == 0);
    free(output);
    free_buf(buf);
}

void test_string_maybe_deserialize_invalid() {
    printf("  test_string_maybe_deserialize_invalid\n");
    Buffer* input = buf(8);
    memcpy(input->data, "XXXXXXXX", 8);
    char* correct = NULL;
    char* output = maybe_deserialize_string(input);
    assert(output == correct);
    free_buf(input);
}

void test_matrix_roundtrip() {
    printf("  test_matrix_roundtrip\n");
    float data[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix input = mat_from_array(2, 3, data);
    Matrix correct = input;
    Buffer* buf = serialize_matrix(input);
    Matrix output = maybe_deserialize_matrix(buf);
    assert(eq(output, correct));
    free_mat(input);
    free_mat(output);
    free_buf(buf);
}

void test_matrix_roundtrip_non_contiguous() {
    printf("  test_matrix_roundtrip_non_contiguous\n");
    float data[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix input = mat_from_array(2, 3, data);
    Matrix transposed = transpose(input);
    Matrix correct = transposed;
    Buffer* buf = serialize_matrix(transposed);
    Matrix output = maybe_deserialize_matrix(buf);
    assert(eq(output, correct));
    free_mat(input);
    free_mat(output);
    free_buf(buf);
}

void test_matrix_maybe_deserialize_invalid() {
    printf("  test_matrix_maybe_deserialize_invalid\n");
    Buffer* input = buf(8);
    memcpy(input->data, "XXXXXXXX", 8);
    Matrix correct = empty(0, 0);
    Matrix output = maybe_deserialize_matrix(input);
    assert(output.rows == correct.rows);
    assert(output.cols == correct.cols);
    free_buf(input);
    free_mat(output);
}

int main() {
    printf("codec tests:\n");
    test_string_roundtrip();
    test_string_maybe_deserialize_invalid();
    test_matrix_roundtrip();
    test_matrix_roundtrip_non_contiguous();
    test_matrix_maybe_deserialize_invalid();
    return 0;
}

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

void test_string_maybe_deserialize_unterminated() {
    printf("  test_string_maybe_deserialize_unterminated\n");
    Buffer* input = buf(6);
    memcpy(input->data, "STRabc", 6);
    char* output = maybe_deserialize_string(input);
    assert(output == NULL);
    free_buf(input);
}

void test_packet_roundtrip() {
    printf("  test_packet_roundtrip\n");
    float data[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix mat = mat_from_array(2, 3, data);
    Packet input = { .matrix = mat, .token_pos = 42 };
    Buffer* buf = serialize_packet(input);
    Packet output = maybe_deserialize_packet(buf);
    assert(eq(output.matrix, input.matrix));
    assert(output.token_pos == 42);
    free_mat(mat);
    free_mat(output.matrix);
    free_buf(buf);
}

void test_packet_roundtrip_non_contiguous() {
    printf("  test_packet_roundtrip_non_contiguous\n");
    float data[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix mat = mat_from_array(2, 3, data);
    Matrix transposed = transpose(mat);
    Packet input = { .matrix = transposed, .token_pos = 7 };
    Buffer* buf = serialize_packet(input);
    Packet output = maybe_deserialize_packet(buf);
    assert(eq(output.matrix, transposed));
    assert(output.token_pos == 7);
    free_mat(mat);
    free_mat(output.matrix);
    free_buf(buf);
}

void test_packet_maybe_deserialize_invalid() {
    printf("  test_packet_maybe_deserialize_invalid\n");
    Buffer* input = buf(8);
    memcpy(input->data, "XXXXXXXX", 8);
    Packet output = maybe_deserialize_packet(input);
    assert(output.matrix.rows == 0);
    assert(output.matrix.cols == 0);
    assert(output.token_pos == 0);
    free_buf(input);
    free_mat(output.matrix);
}

int main() {
    printf("codec tests:\n");
    test_string_roundtrip();
    test_string_maybe_deserialize_invalid();
    test_string_maybe_deserialize_unterminated();
    test_packet_roundtrip();
    test_packet_roundtrip_non_contiguous();
    test_packet_maybe_deserialize_invalid();
    return 0;
}

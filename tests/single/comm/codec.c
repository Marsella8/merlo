#include <string.h>

#include "codec.h"
#include "rpi.h"

static void test_string_roundtrip(void) {
    printk("  test_string_roundtrip\n");
    const char* input = "Church";
    const char* correct = input;
    Buffer* buf = serialize_string((char*)input);
    char* actual = maybe_deserialize_string(buf);
    assert(actual != NULL);
    assert(strcmp(actual, correct) == 0);
    free(actual);
    free_buf(buf);
}

static void test_string_maybe_deserialize_invalid(void) {
    printk("  test_string_maybe_deserialize_invalid\n");
    Buffer* input = buf(8);
    memcpy(input->data, "XXXXXXXX", 8);
    char* correct = NULL;
    char* actual = maybe_deserialize_string(input);
    assert(actual == correct);
    free_buf(input);
}

static void test_packet_roundtrip(void) {
    printk("  test_packet_roundtrip\n");
    float data[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix mat = mat_from_array(2, 3, data);
    Packet input = { .matrix = mat, .token_pos = 42 };
    Buffer* buf = serialize_packet(input);
    Packet actual = maybe_deserialize_packet(buf);
    assert(eq(actual.matrix, input.matrix));
    assert(actual.token_pos == 42);
    free_mat(mat);
    free_mat(actual.matrix);
    free_buf(buf);
}

static void test_packet_roundtrip_non_contiguous(void) {
    printk("  test_packet_roundtrip_non_contiguous\n");
    float data[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix mat = mat_from_array(2, 3, data);
    Matrix transposed = transpose(mat);
    Packet input = { .matrix = transposed, .token_pos = 7 };
    Buffer* buf = serialize_packet(input);
    Packet actual = maybe_deserialize_packet(buf);
    assert(eq(actual.matrix, transposed));
    assert(actual.token_pos == 7);
    free_mat(mat);
    free_mat(actual.matrix);
    free_buf(buf);
}

static void test_packet_maybe_deserialize_invalid(void) {
    printk("  test_packet_maybe_deserialize_invalid\n");
    Buffer* input = buf(8);
    memcpy(input->data, "XXXXXXXX", 8);
    Packet actual = maybe_deserialize_packet(input);
    assert(actual.matrix.rows == 0);
    assert(actual.matrix.cols == 0);
    assert(actual.token_pos == 0);
    free_buf(input);
    free_mat(actual.matrix);
}

void notmain(void) {
    kmalloc_init();
    printk("codec tests:\n");
    test_string_roundtrip();
    test_string_maybe_deserialize_invalid();
    test_packet_roundtrip();
    test_packet_roundtrip_non_contiguous();
    test_packet_maybe_deserialize_invalid();
    printk("tests/single/comm/codec: pass\n");
    clean_reboot();
}

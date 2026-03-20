#include <string.h>

#include "codec.h"
#include "matrix.h"
#include "utils.h"

const char* PKT_ID = "PKT";

static inline char* insert(char* ptr, const void* src, size_t size) {
    memcpy(ptr, src, size);
    return ptr + size;
}

static inline char* extract(char* ptr, void* dst, size_t size) {
    memcpy(dst, ptr, size);
    return ptr + size;
}

Buffer* serialize_packet(Packet pkt) {
    Matrix c = as_contiguous(pkt.matrix);
    size_t data_len = num_bytes(c);
    size_t packet_len = strlen(PKT_ID) + sizeof(size_t) * 3 + sizeof(int) * 2 + data_len;
    Buffer* b = buf(packet_len);
    char* ptr = b->data;
    ptr = insert(ptr, PKT_ID, strlen(PKT_ID));
    ptr = insert(ptr, &pkt.token_pos, sizeof(size_t));
    ptr = insert(ptr, &c.rows, sizeof(size_t));
    ptr = insert(ptr, &c.cols, sizeof(size_t));
    ptr = insert(ptr, &c.row_stride, sizeof(int));
    ptr = insert(ptr, &c.col_stride, sizeof(int));
    memcpy(ptr, c.buffer->data, data_len);
    free_mat(c);
    return b;
}

Packet maybe_deserialize_packet(Buffer* buffer) {
    size_t header_len = strlen(PKT_ID) + sizeof(size_t) * 3 + sizeof(int) * 2;
    if (buffer->size < header_len || memcmp(buffer->data, PKT_ID, strlen(PKT_ID)) != 0) {
        return (Packet){ .matrix = empty(0, 0), .token_pos = 0 };
    }
    char* ptr = buffer->data + strlen(PKT_ID);
    size_t token_pos, rows, cols;
    int row_stride, col_stride;
    ptr = extract(ptr, &token_pos, sizeof(size_t));
    ptr = extract(ptr, &rows, sizeof(size_t));
    ptr = extract(ptr, &cols, sizeof(size_t));
    ptr = extract(ptr, &row_stride, sizeof(int));
    ptr = extract(ptr, &col_stride, sizeof(int));
    size_t data_len = num_bytes((Matrix){ .rows = rows, .cols = cols });
    Buffer* b = buf(data_len);
    memcpy(b->data, ptr, data_len);
    return (Packet){
        .matrix =
            {
                .buffer = b,
                .rows = rows,
                .cols = cols,
                .row_stride = row_stride,
                .col_stride = col_stride,
                .offset = 0,
                .owned = true,
            },
        .token_pos = token_pos,
    };
}

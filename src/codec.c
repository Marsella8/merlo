#include <stdlib.h>
#include <string.h>

#include "codec.h"
#include "matrix.h"

char* STRING_ID = "STR";

Buffer* serialize_string(char* string) {
    size_t len = strlen(string) + 1;
    size_t packet_len = strlen(STRING_ID) + len;
    Buffer* b = buf(packet_len);
    char* ptr = b->data;
    
    memcpy(ptr, STRING_ID, strlen(STRING_ID));
    ptr += strlen(STRING_ID);
    
    memcpy(ptr, string, len);
    
    return b;
}

char* maybe_deserialize_string(Buffer* buf) {
    if (memcmp(buf->data, STRING_ID, strlen(STRING_ID)) != 0) {
        return NULL;
    }
    char* ptr = buf->data + strlen(STRING_ID);
    size_t len = strnlen(ptr, buf->size - strlen(STRING_ID)) + 1;
    char* data = malloc(len);
    memcpy(data, ptr, len);
    return data;
}

const char* PKT_ID = "PKT";

Buffer* serialize_packet(Packet pkt) {
    Matrix c = as_contiguous(pkt.matrix);
    size_t data_len = num_bytes(c);
    size_t packet_len = strlen(PKT_ID) + sizeof(size_t) + sizeof(size_t) + sizeof(size_t) + sizeof(int) + sizeof(int) + data_len;
    Buffer* b = buf(packet_len);
    char* ptr = b->data;
    memcpy(ptr, PKT_ID, strlen(PKT_ID));
    ptr += strlen(PKT_ID);
    memcpy(ptr, &pkt.token_pos, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, &c.rows, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, &c.cols, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, &c.row_stride, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &c.col_stride, sizeof(int));
    ptr += sizeof(int);
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
    memcpy(&token_pos, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(&rows, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(&cols, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(&row_stride, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&col_stride, ptr, sizeof(int));
    ptr += sizeof(int);
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


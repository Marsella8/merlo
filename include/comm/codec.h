#ifndef CODEC_H
#define CODEC_H

#include "matrix.h"

typedef struct {
    Matrix matrix;
    size_t token_pos;
} Packet;

Buffer* serialize_packet(Packet pkt);
Packet maybe_deserialize_packet(Buffer* buf);

#endif

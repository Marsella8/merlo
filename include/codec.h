#ifndef CODEC_H
#define CODEC_H

#include "matrix.h"

Buffer* serialize_string(char* string);
char* maybe_deserialize_string(Buffer* buf);
Buffer* serialize_matrix(Matrix mat);
Matrix maybe_deserialize_matrix(Buffer* buffer);

#endif

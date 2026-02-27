#ifndef SAMPLE_H
#define SAMPLE_H

#include "matrix.h"
#include "model.h"

size_t argmax(Matrix logits);
size_t sample(Matrix logits, float temperature);

#endif // SAMPLE_H

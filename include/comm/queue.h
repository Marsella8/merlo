#ifndef QUEUE_H
#define QUEUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "codec.h"
#include "matrix.h"

#define INT_QUEUE_CAPACITY 100
#define PACKET_QUEUE_CAPACITY 100
#define STRING_QUEUE_CAPACITY 100
#define BYTE_QUEUE_CAPACITY (64 * 1024)

#define QUEUE_DECL_SIZED(T, Name, lower, CAP)                                \
typedef struct {                                                             \
    T elems[CAP];                                                            \
    size_t head;                                                             \
    size_t tail;                                                             \
    size_t len;                                                              \
} Name;                                                                      \
                                                                             \
static inline Name lower##_init(void) {                                      \
    return (Name){0};                                                        \
}                                                                            \
                                                                             \
static inline size_t lower##_capacity(void) {                                \
    return (CAP);                                                            \
}                                                                            \
                                                                             \
static inline size_t lower##_len(const Name *q) {                            \
    return q->len;                                                           \
}                                                                            \
                                                                             \
static inline bool lower##_is_empty(const Name *q) {                         \
    return q->len == 0;                                                      \
}                                                                            \
                                                                             \
static inline bool lower##_is_full(const Name *q) {                          \
    return q->len == (CAP);                                                  \
}                                                                            \
                                                                             \
static inline void lower##_enqueue(Name *q, T elem) {                        \
    assert(!lower##_is_full(q));                                             \
    q->elems[q->tail] = elem;                                                \
    q->tail = (q->tail + 1) % (CAP);                                         \
    q->len++;                                                                \
}                                                                            \
                                                                             \
static inline T lower##_deque(Name *q) {                                     \
    assert(!lower##_is_empty(q));                                            \
    T elem = q->elems[q->head];                                              \
    q->head = (q->head + 1) % (CAP);                                         \
    q->len--;                                                                \
    return elem;                                                             \
}                                                                            \
                                                                             \
static inline T lower##_peek(const Name *q, size_t k) {                      \
    assert(k < q->len);                                                      \
    return q->elems[(q->head + k) % (CAP)];                                  \
}

QUEUE_DECL_SIZED(int, IntQueue, int_queue, INT_QUEUE_CAPACITY)
QUEUE_DECL_SIZED(Packet, PacketQueue, packet_queue, PACKET_QUEUE_CAPACITY)
QUEUE_DECL_SIZED(char *, StringQueue, string_queue, STRING_QUEUE_CAPACITY)
QUEUE_DECL_SIZED(uint8_t, ByteQueue, byte_queue, BYTE_QUEUE_CAPACITY)

#endif

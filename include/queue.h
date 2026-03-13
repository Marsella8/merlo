#ifndef QUEUE_H
#define QUEUE_H

#include <stdbool.h>
#include <stddef.h>

#include "codec.h"
#include "matrix.h"

#define QUEUE_SIZE 100

#define QUEUE_DECL(T, Name, lower)                                           \
typedef struct {                                                             \
    T elems[QUEUE_SIZE];                                                     \
    size_t head;                                                             \
    size_t tail;                                                             \
} Name;                                                                      \
                                                                             \
static inline Name lower##_init(void) {                                      \
    return (Name){0};                                                        \
}                                                                            \
                                                                             \
static inline void lower##_enqueue(Name *q, T elem) {                         \
    q->elems[q->tail] = elem;                                                \
    q->tail = (q->tail + 1) % QUEUE_SIZE;                                    \
}                                                                            \
                                                                             \
static inline T lower##_deque(Name *q) {                                      \
    T elem = q->elems[q->head];                                              \
    q->head = (q->head + 1) % QUEUE_SIZE;                                    \
    return elem;                                                             \
}                                                                            \
                                                                             \
static inline bool lower##_is_empty(Name q) {                                 \
    return q.head == q.tail;                                                 \
}

QUEUE_DECL(int, IntQueue, int_queue)
QUEUE_DECL(Packet, PacketQueue, packet_queue)
QUEUE_DECL(char *, StringQueue, string_queue)

#endif

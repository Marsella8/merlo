#include <assert.h>
#include <stdio.h>

#include "queue.h"

void test_queue_init() {
    printf("  test_queue_init\n");
    IntQueue queue = int_queue_init();

    assert(queue.head == 0);
    assert(queue.tail == 0);
    assert(int_queue_is_empty(queue));
}

void test_enqueue_deque() {
    printf("  test_enqueue_deque\n");
    IntQueue queue = int_queue_init();

    int_queue_enqueue(&queue, 1);
    int_queue_enqueue(&queue, 2);
    int_queue_enqueue(&queue, 3);

    assert(!int_queue_is_empty(queue));
    assert(int_queue_deque(&queue) == 1);
    assert(int_queue_deque(&queue) == 2);
    assert(int_queue_deque(&queue) == 3);
    assert(int_queue_is_empty(queue));
}

void test_wraparound() {
    printf("  test_wraparound\n");
    IntQueue queue = int_queue_init();

    for (int i = 0; i < QUEUE_SIZE - 1; i++) {
        int_queue_enqueue(&queue, i);
    }

    for (int i = 0; i < 10; i++) {
        assert(int_queue_deque(&queue) == i);
    }

    for (int i = 0; i < 10; i++) {
        int_queue_enqueue(&queue, QUEUE_SIZE - 1 + i);
    }

    for (int i = 10; i < QUEUE_SIZE - 1 + 10; i++) {
        assert(int_queue_deque(&queue) == i);
    }

    assert(int_queue_is_empty(queue));
}

int main() {
    printf("queue tests:\n");
    test_queue_init();
    test_enqueue_deque();
    test_wraparound();
    return 0;
}

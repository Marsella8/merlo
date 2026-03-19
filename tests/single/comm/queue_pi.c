#include "queue.h"
#include "rpi.h"

static void test_queue_init(void) {
    printk("  test_queue_init\n");
    IntQueue queue = int_queue_init();

    assert(queue.head == 0);
    assert(queue.tail == 0);
    assert(queue.len == 0);
    assert(int_queue_is_empty(&queue));
}

static void test_enqueue_deque(void) {
    printk("  test_enqueue_deque\n");
    IntQueue queue = int_queue_init();

    int_queue_enqueue(&queue, 1);
    int_queue_enqueue(&queue, 2);
    int_queue_enqueue(&queue, 3);

    assert(int_queue_len(&queue) == 3);
    assert(!int_queue_is_empty(&queue));
    assert(int_queue_deque(&queue) == 1);
    assert(int_queue_deque(&queue) == 2);
    assert(int_queue_deque(&queue) == 3);
    assert(int_queue_is_empty(&queue));
}

static void test_peek(void) {
    printk("  test_peek\n");
    IntQueue queue = int_queue_init();

    int_queue_enqueue(&queue, 10);
    int_queue_enqueue(&queue, 20);
    int_queue_enqueue(&queue, 30);

    assert(int_queue_peek(&queue, 0) == 10);
    assert(int_queue_peek(&queue, 1) == 20);
    assert(int_queue_peek(&queue, 2) == 30);
    assert(int_queue_len(&queue) == 3);
}

static void test_wraparound(void) {
    printk("  test_wraparound\n");
    IntQueue queue = int_queue_init();

    for (int i = 0; i < (int)int_queue_capacity(); i++) {
        int_queue_enqueue(&queue, i);
    }

    for (int i = 0; i < 10; i++) {
        assert(int_queue_deque(&queue) == i);
    }

    for (int i = 0; i < 10; i++) {
        int_queue_enqueue(&queue, (int)int_queue_capacity() + i);
    }

    for (int i = 10; i < (int)int_queue_capacity() + 10; i++) {
        assert(int_queue_deque(&queue) == i);
    }

    assert(int_queue_is_empty(&queue));
}

void notmain(void) {
    printk("single/comm queue tests:\n");
    test_queue_init();
    test_enqueue_deque();
    test_peek();
    test_wraparound();
    printk("single/comm/queue_pi: pass\n");
    clean_reboot();
}

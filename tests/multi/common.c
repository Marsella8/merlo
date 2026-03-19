#include <stddef.h>
#include <string.h>

#include "rpi.h"

#include "common.h"

const char comm_test_ping[] = "comm-ping";
const char comm_test_pong[] = "comm-pong";

void comm_test_init(void) {
    kmalloc_init();
    comm_init();
    assert(comm_add_recv(COMM_TEST_RX_PIN));
    comm_init_send(COMM_TEST_TX_PIN);
}

Buffer comm_test_static_buffer(const char *payload) {
    return (Buffer){
        .data = (void *)payload,
        .size = strlen(payload),
    };
}

bool comm_test_payload_eq(const Buffer *frame, const char *payload) {
    size_t n = strlen(payload);
    return frame->size == n && memcmp(frame->data, payload, n) == 0;
}

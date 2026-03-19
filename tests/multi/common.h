#ifndef TESTS_MULTI_COMMON_H
#define TESTS_MULTI_COMMON_H

#include <stdbool.h>

#include "comm.h"

enum {
    COMM_TEST_RX_PIN = 21,
    COMM_TEST_TX_PIN = 20,
    COMM_TEST_RETRY_DELAY_MS = 200,
};

extern const char comm_test_ping[];
extern const char comm_test_pong[];

void comm_test_init(void);
Buffer comm_test_static_buffer(const char *payload);
bool comm_test_payload_eq(const Buffer *frame, const char *payload);

#endif

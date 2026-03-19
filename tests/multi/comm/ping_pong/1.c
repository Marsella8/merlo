#include "rpi.h"

#include "../../common.h"

void notmain(void) {
    Buffer pong_frame = comm_test_static_buffer(comm_test_pong);

    printk("multi/comm/ping_pong/1: init tx=%d rx=%d\n",
           COMM_TEST_TX_PIN,
           COMM_TEST_RX_PIN);
    comm_test_init();

    for (;;) {
        Buffer *frame = comm_try_recv_frame(COMM_TEST_RX_PIN);
        if (frame == NULL) {
            continue;
        }

        bool ok = comm_test_payload_eq(frame, comm_test_ping);
        free_buf(frame);

        if (!ok) {
            continue;
        }

        printk("multi/comm/ping_pong/1: send pong\n");
        assert(comm_send(COMM_TEST_TX_PIN, &pong_frame));
        delay_ms(50);
        printk("multi/comm/ping_pong/1: pass\n");
        clean_reboot();
    }
}

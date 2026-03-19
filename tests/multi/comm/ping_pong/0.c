#include "rpi.h"

#include "../../common.h"

void notmain(void) {
    Buffer ping_frame = comm_test_static_buffer(comm_test_ping);

    printk("multi/comm/ping_pong/0: init tx=%d rx=%d\n",
           COMM_TEST_TX_PIN,
           COMM_TEST_RX_PIN);
    comm_test_init();

    for (;;) {
        printk("multi/comm/ping_pong/0: send ping\n");
        assert(comm_send(COMM_TEST_TX_PIN, &ping_frame));
        delay_ms(COMM_TEST_RETRY_DELAY_MS);

        Buffer *frame = comm_try_recv_frame(COMM_TEST_RX_PIN);
        if (frame == NULL) {
            continue;
        }

        bool ok = comm_test_payload_eq(frame, comm_test_pong);
        free_buf(frame);

        if (ok) {
            printk("multi/comm/ping_pong/0: pass\n");
            clean_reboot();
        }
    }
}

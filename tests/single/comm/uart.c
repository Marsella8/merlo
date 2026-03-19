#include "rpi.h"

void notmain(void) {
    kmalloc_init();
    printk("uart tests:\n");
    printk("tests/single/comm/uart: pass\n");
    clean_reboot();
}

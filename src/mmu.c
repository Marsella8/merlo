#include "rpi.h"

static uint32_t pt[4096] __attribute__((aligned(16384)));

void enable_dcache(void) {
    for (unsigned i = 0; i < 4096; i++)
        pt[i] = (i << 20) | (i < 0x200 ? 0x1C0E : 0x0C02);

    uint32_t zero = 0;
    asm volatile(
        "MCR p15, 0, %[pt],   c2, c0, 0 \n" // TTBR0 = page table
        "MCR p15, 0, %[dom],  c3, c0, 0 \n" // DACR: domain 0 = client
        "MCR p15, 0, %[zero], c8, c7, 0 \n" // invalidate TLB
        "MCR p15, 0, %[zero], c7, c10, 4\n" // DSB
        :: [pt]"r"(pt), [dom]"r"(1u), [zero]"r"(zero)
    );

    unsigned sctlr;
    asm volatile("MRC p15, 0, %0, c1, c0, 0" : "=r"(sctlr));
    sctlr |= (1<<0)|(1<<2)|(1<<11)|(1<<12);   // MMU, D-cache, BTC, I-cache
    asm volatile(
        "MCR p15, 0, %[zero], c7, c10, 4\n" // DSB
        "MCR p15, 0, %[sctlr],c1, c0, 0\n" // enable
        "MCR p15, 0, %[zero], c7, c5,  4\n" // prefetch flush
        :: [sctlr]"r"(sctlr), [zero]"r"(zero)
    );
}

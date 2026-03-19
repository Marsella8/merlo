#include <stdbool.h>
#include <string.h>

#include "kernels/gpu.h"
#include "rpi.h"

void cache_flush_all(void);

#define MBOX_BASE   0x2000B880
#define MBOX_READ   (*(volatile uint32_t *)(MBOX_BASE + 0x00))
#define MBOX_STATUS (*(volatile uint32_t *)(MBOX_BASE + 0x18))
#define MBOX_WRITE  (*(volatile uint32_t *)(MBOX_BASE + 0x20))
#define MBOX_FULL   0x80000000
#define MBOX_CH     8

#define V3D_BASE    0x20C00000
#define V3D_L2CACTL (V3D_BASE + 0x020)
#define V3D_SLCACTL (V3D_BASE + 0x024)
#define V3D_SRQPC   (V3D_BASE + 0x430)
#define V3D_SRQUA   (V3D_BASE + 0x434)
#define V3D_SRQCS   (V3D_BASE + 0x43c)
#define V3D_DBCFG   (V3D_BASE + 0xe00)
#define V3D_DBQITE  (V3D_BASE + 0xe2c)
#define V3D_DBQITC  (V3D_BASE + 0xe30)

static bool gpu_init_done = false;

uint32_t gpu_bus_addr(const void *ptr) {
    return GPU_BUS_ALIAS + (uint32_t)(uintptr_t)ptr;
}

uint32_t gpu_bits_from_float(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint16_t gpu_fp32_to_bf16(float value) {
    uint32_t bits = gpu_bits_from_float(value);
    return (uint16_t)(bits >> 16);
}

unsigned gpu_pad16(unsigned n) {
    return (n + 15u) & ~15u;
}

unsigned gpu_pad_qpu(unsigned n) {
    unsigned grain = GPU_NUM_QPUS * 16u;
    return ((n + grain - 1u) / grain) * grain;
}

float *gpu_matrix_data(Matrix x) {
    return (float *)x.buffer->data;
}

void gpu_copy_matrix(Matrix src, float *dst) {
    for (size_t r = 0; r < src.rows; r++) {
        for (size_t c = 0; c < src.cols; c++) {
            *dst++ = *at(src, r, c);
        }
    }
}

void gpu_copy_array_into_matrix(const float *src, Matrix dst) {
    for (size_t r = 0; r < dst.rows; r++) {
        for (size_t c = 0; c < dst.cols; c++) {
            *at(dst, r, c) = *src++;
        }
    }
}

void gpu_copy_matrix_rows_padded(Matrix src, float *dst, unsigned stride, float pad_value) {
    for (size_t r = 0; r < src.rows; r++) {
        unsigned c = 0;
        for (; c < src.cols; c++) {
            dst[r * stride + c] = *at(src, r, c);
        }
        for (; c < stride; c++) {
            dst[r * stride + c] = pad_value;
        }
    }
}

void gpu_copy_padded_rows_into_matrix(const float *src, unsigned stride, Matrix dst) {
    for (size_t r = 0; r < dst.rows; r++) {
        for (size_t c = 0; c < dst.cols; c++) {
            *at(dst, r, c) = src[r * stride + c];
        }
    }
}

void gpu_fp32_to_panel_layout(const float *src, float *panel, unsigned rows, unsigned cols) {
    for (unsigned p = 0; p < rows / 16u; p++) {
        for (unsigned col = 0; col < cols; col++) {
            for (unsigned lane = 0; lane < 16u; lane++) {
                panel[(p * cols + col) * 16u + lane] = src[(p * 16u + lane) * cols + col];
            }
        }
    }
}

void gpu_fp32_from_panel_layout(const float *panel, float *dst, unsigned rows, unsigned cols) {
    for (unsigned p = 0; p < rows / 16u; p++) {
        for (unsigned col = 0; col < cols; col++) {
            for (unsigned lane = 0; lane < 16u; lane++) {
                dst[(p * 16u + lane) * cols + col] = panel[(p * cols + col) * 16u + lane];
            }
        }
    }
}

void gpu_bf16_to_panel_layout(const uint16_t *src, uint16_t *panel, unsigned rows, unsigned cols) {
    for (unsigned p = 0; p < rows / 16u; p++) {
        for (unsigned col = 0; col < cols; col++) {
            for (unsigned lane = 0; lane < 16u; lane++) {
                panel[p * cols * 16u + col * 16u + lane] = src[(p * 16u + lane) * cols + col];
            }
        }
    }
}

static void gpu_property(uint32_t *msg) {
    cache_flush_all();
    while (MBOX_STATUS & MBOX_FULL) {}
    MBOX_WRITE = ((uint32_t)msg & ~0xFu) | MBOX_CH;
    while ((MBOX_READ & 0xF) != MBOX_CH) {}
    cache_flush_all();
}

static void gpu_qpu_enable(void) {
    uint32_t p[7] __attribute__((aligned(16))) = {
        28, 0, 0x30012, 4, 4, 1, 0
    };
    gpu_property(p);
}

void gpu_init(void) {
    if (gpu_init_done)
        return;
    gpu_qpu_enable();
    gpu_init_done = true;
}

void gpu_exec_direct(uint32_t code, uint32_t uniform_ptrs[], unsigned num_qpus) {
    gpu_init();
    cache_flush_all();

    PUT32(V3D_DBCFG, 0);
    PUT32(V3D_DBQITE, 0);
    PUT32(V3D_DBQITC, (uint32_t)-1);
    PUT32(V3D_L2CACTL, 1 << 2);
    PUT32(V3D_SLCACTL, (uint32_t)-1);
    PUT32(V3D_SRQCS, (1 << 7) | (1 << 8) | (1 << 16));

    for (unsigned q = 0; q < num_qpus; q++) {
        PUT32(V3D_SRQUA, uniform_ptrs[q]);
        PUT32(V3D_SRQPC, code);
    }

    while (((GET32(V3D_SRQCS) >> 16) & 0xff) != num_qpus) {}

    PUT32(V3D_L2CACTL, 1 << 2);
    PUT32(V3D_SLCACTL, (uint32_t)-1);
    cache_flush_all();
}

SHELL := /bin/bash

ROOT_DIR := /home/pietro/Documents/Code/merlo
BUILD_DIR := $(ROOT_DIR)/build

ARM := arm-none-eabi
CC := $(ARM)-gcc
OCP := $(ARM)-objcopy

LIBPI_DIR := $(ROOT_DIR)/libpi
START := $(LIBPI_DIR)/staff-start.o
LDLIBS := $(LIBPI_DIR)/libpi.a -lm -lgcc
MAKE_BIN ?= /usr/bin/make
PI_INSTALL := /home/pietro/Documents/cs140e-26win/bin/pi-install.linux
PI0_DEVICE := /dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_ce05c88fa13cef11a1e7b16ed981d5d9-if00-port0
PI1_DEVICE := /dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_384bb796a13cef11b18cba6ed981d5d9-if00-port0

INCLUDES := -I$(ROOT_DIR)/include -I$(ROOT_DIR)/include/comm -I$(ROOT_DIR)/include/nn -I$(ROOT_DIR)/include/model -I$(ROOT_DIR)/include/kernels -I$(LIBPI_DIR)/include -I$(LIBPI_DIR) -I$(LIBPI_DIR)/src -I$(LIBPI_DIR)/libc
ARCH_FLAGS := -ffreestanding -nostdlib -nostartfiles -mcpu=arm1176jzf-s -mtune=arm1176jzf-s -mno-unaligned-access -mtp=soft
WARN_FLAGS := -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-variable
COMMON_CFLAGS := -D__RPI__ -std=gnu99 $(ARCH_FLAGS) $(WARN_FLAGS) $(INCLUDES) -O3
CFLAGS := $(COMMON_CFLAGS)
LDFLAGS := $(ARCH_FLAGS) -T $(LIBPI_DIR)/memmap

COMMON_SRCS := \
	$(wildcard $(ROOT_DIR)/src/comm/*.c) \
	$(wildcard $(ROOT_DIR)/src/model/*.c) \
	$(wildcard $(ROOT_DIR)/src/nn/*.c) \
	$(wildcard $(ROOT_DIR)/src/kernels/*.c) \
	$(wildcard $(ROOT_DIR)/src/kernels/*/*.c) \
	$(ROOT_DIR)/src/utils.c \
	$(ROOT_DIR)/src/mmu.c

# Rebuild when headers change (gcc -MMD is not used for these one-shot links).
MERLO_HDRS := $(wildcard $(ROOT_DIR)/include/*.h) \
	$(wildcard $(ROOT_DIR)/include/*/*.h) \
	$(wildcard $(ROOT_DIR)/include/*/*/*.h)
LIBPI_HDRS := $(wildcard $(LIBPI_DIR)/include/*.h)

HEAD_ELF := $(BUILD_DIR)/head.elf
HEAD_BIN := $(BUILD_DIR)/head.bin
PIPE_ELF := $(BUILD_DIR)/pipe.elf
PIPE_BIN := $(BUILD_DIR)/pipe.bin

.PHONY: clean libpi head head-bin pipe pipe-bin

$(HEAD_ELF): $(ROOT_DIR)/src/head.c $(COMMON_SRCS) $(MERLO_HDRS) $(LIBPI_HDRS) | libpi
	mkdir -p $(dir $@)
	$(CC) $(COMMON_CFLAGS) $(LDFLAGS) -o $@ $(START) $(ROOT_DIR)/src/head.c $(COMMON_SRCS) $(LDLIBS)

$(PIPE_ELF): $(ROOT_DIR)/src/pipe.c $(COMMON_SRCS) $(MERLO_HDRS) $(LIBPI_HDRS) | libpi
	mkdir -p $(dir $@)
	$(CC) $(COMMON_CFLAGS) $(LDFLAGS) -o $@ $(START) $(ROOT_DIR)/src/pipe.c $(COMMON_SRCS) $(LDLIBS)

$(HEAD_BIN): $(HEAD_ELF)
	$(OCP) $< -O binary $@

$(PIPE_BIN): $(PIPE_ELF)
	$(OCP) $< -O binary $@

head-bin: $(HEAD_BIN)

pipe-bin: $(PIPE_BIN)

head: $(HEAD_BIN)
	$(PI_INSTALL) --device $(PI0_DEVICE) $<

pipe: $(PIPE_BIN)
	$(PI_INSTALL) --device $(PI1_DEVICE) $<

clean:
	rm -rf $(ROOT_DIR)/build $(ROOT_DIR)/tests/build

libpi:
	$(MAKE_BIN) -C $(LIBPI_DIR) CS140E_2026_PATH=$(ROOT_DIR) PROGS=

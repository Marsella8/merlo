ARM = arm-none-eabi
CC = $(ARM)-gcc
OD = $(ARM)-objdump
OCP = $(ARM)-objcopy

DEVICE_NAME ?= HEAD
LIBPI_DIR = libpi
LIBPI_ENV = CS140E_2026_PATH=$(abspath .)

BUILD_DIR = build
PROGRAM = merlo
ELF = $(BUILD_DIR)/$(PROGRAM).elf
BIN = $(BUILD_DIR)/$(PROGRAM).bin
LIST = $(BUILD_DIR)/$(PROGRAM).list

INCLUDES = -Iinclude -Iinclude/comm -Iinclude/nn -Iinclude/model -I$(LIBPI_DIR)/include -I$(LIBPI_DIR) -I$(LIBPI_DIR)/src -I$(LIBPI_DIR)/libc
ARCH_FLAGS = -ffreestanding -nostdlib -nostartfiles -mcpu=arm1176jzf-s -mtune=arm1176jzf-s -mno-unaligned-access -mtp=soft
WARN_FLAGS = -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-variable
COMMON_FLAGS = -D__RPI__ -DDEVICE_NAME=\"$(DEVICE_NAME)\" -std=gnu99 $(ARCH_FLAGS) $(WARN_FLAGS) $(INCLUDES)
OPT_FLAGS ?= -O3
CFLAGS = $(COMMON_FLAGS) $(OPT_FLAGS)
LDFLAGS = $(ARCH_FLAGS) -T $(LIBPI_DIR)/memmap
LDLIBS = $(LIBPI_DIR)/libpi.a -lm -lgcc
START = $(LIBPI_DIR)/staff-start.o

SRCS = $(shell find src -name '*.c' | sort)
OBJS = $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SRCS))

.PHONY: all clean debug release libpi libpi-clean

all: $(BIN) $(LIST)

debug: OPT_FLAGS = -O0 -g -DSAFETY
debug: clean $(BIN) $(LIST)

release: OPT_FLAGS = -O3
release: clean $(BIN) $(LIST)

libpi:
	$(MAKE) -C $(LIBPI_DIR) $(LIBPI_ENV) PROGS=

libpi-clean:
	$(MAKE) -C $(LIBPI_DIR) $(LIBPI_ENV) clean

clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(ELF): $(OBJS) | $(BUILD_DIR) libpi
	$(CC) $(LDFLAGS) -o $@ $(START) $(OBJS) $(LDLIBS)

$(BIN): $(ELF)
	$(OCP) $< -O binary $@

$(LIST): $(ELF)
	$(OD) -d $< > $@

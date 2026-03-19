SHELL := /bin/bash

ROOT_DIR := /home/pietro/Documents/Code/merlo

PI_INSTALL := /home/pietro/Documents/cs140e-26win/bin/pi-install.linux
PI0_DEVICE := /dev/ttyUSB0
PI1_DEVICE := /dev/ttyUSB1
PI2_DEVICE := /dev/ttyUSB2
PI3_DEVICE := /dev/ttyUSB3

TEST_OUT := $(ROOT_DIR)/tests/build

include $(ROOT_DIR)/Makefile

# Multi-test shared header (harmless extra dep for single tests).
TEST_HDRS := $(MERLO_HDRS) $(LIBPI_HDRS) $(ROOT_DIR)/tests/multi/common.h

.PHONY: build test singlebuild singletest multibuild multitest

$(TEST_OUT)/%.elf: $(ROOT_DIR)/%.c $(TEST_SRCS) $(TEST_HDRS) | libpi
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(START) $< $(TEST_SRCS) $(LDLIBS)

$(TEST_OUT)/%.bin: $(TEST_OUT)/%.elf
	$(OCP) $< -O binary $@

build: $(addprefix $(TEST_OUT)/,$(addsuffix .bin,$(TESTS)))

ifdef MULTI_TEST_DIRS
test: build
	@for testdir in $(MULTI_TEST_DIRS); do \
		$(PI_INSTALL) --device $(PI1_DEVICE) $(TEST_OUT)/$$testdir/1.bin & \
		pid=$$!; \
		sleep 1; \
		$(PI_INSTALL) --device $(PI0_DEVICE) $(TEST_OUT)/$$testdir/0.bin || exit $$?; \
		wait $$pid || exit $$?; \
	done
else
test: build
	@for test in $(TESTS); do \
		$(PI_INSTALL) --device $(TEST_DEVICE) $(TEST_OUT)/$$test.bin || exit $$?; \
	done
endif

singlebuild:
	$(MAKE) -C $(ROOT_DIR)/tests/single build

singletest:
	$(MAKE) -C $(ROOT_DIR)/tests/single test

multibuild:
	$(MAKE) -C $(ROOT_DIR)/tests/multi build

multitest:
	$(MAKE) -C $(ROOT_DIR)/tests/multi test

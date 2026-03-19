ROOT_DIR := /home/pietro/Documents/Code/merlo

PI_TESTS := $(patsubst $(ROOT_DIR)/%.c,%,$(shell find $(ROOT_DIR)/tests/single -name '*.c' | sort))

TESTS := $(PI_TESTS)
TEST_SRCS = $(COMMON_SRCS)
TEST_DEVICE = $(PI1_DEVICE)

include $(ROOT_DIR)/tests/Makefile

.DEFAULT_GOAL := test

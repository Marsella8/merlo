ROOT_DIR := /home/pietro/Documents/Code/merlo

MULTI_TEST_DIRS := $(patsubst $(ROOT_DIR)/%/,%,$(sort $(wildcard $(ROOT_DIR)/tests/multi/*/*/)))
TESTS := $(foreach dir,$(MULTI_TEST_DIRS),$(dir)/0 $(dir)/1)
TEST_SRCS = $(COMMON_SRCS) \
	$(ROOT_DIR)/tests/multi/common.c

include $(ROOT_DIR)/tests/Makefile

.DEFAULT_GOAL := test

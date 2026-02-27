CC = gcc
CFLAGS_COMMON = -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-variable -Iinclude -lm
CFLAGS_RELEASE = $(CFLAGS_COMMON) -O3
CFLAGS_DEBUG = $(CFLAGS_COMMON) -g -O0 -DSAFETY
CFLAGS ?= $(CFLAGS_RELEASE)

SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJS_NO_MAIN = $(filter-out $(BUILD_DIR)/main.o, $(OBJS))

TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS = $(TEST_SRCS:$(TEST_DIR)/%.c=$(BUILD_DIR)/test_%)
MAIN_BIN = $(BUILD_DIR)/main
PERF ?= perf
ARGS ?=

.PHONY: all clean test lint release debug perf

all: $(BUILD_DIR) $(OBJS) $(MAIN_BIN) $(TEST_BINS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

lint:
	clang-tidy $(SRCS) -- $(CFLAGS)

release: CFLAGS = $(CFLAGS_RELEASE)
release: clean all

debug: CFLAGS = $(CFLAGS_DEBUG)
debug: clean all

perf: release
	$(PERF) record -g -m 64 -o perf.data -- $(MAIN_BIN) $(ARGS) && $(PERF) report --stdio --no-children --percent-limit 1 -i perf.data

test: $(TEST_BINS)
	@for test in $(TEST_BINS); do \
		./$$test; \
		if [ $$? -eq 0 ]; then \
			echo "passed!"; \
		else \
			echo "failed!"; \
		fi; \
		echo; \
	done

$(BUILD_DIR)/test_%: $(TEST_DIR)/%.c $(OBJS_NO_MAIN) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS_NO_MAIN) -o $@

$(MAIN_BIN): $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(BUILD_DIR)

CC = gcc
CFLAGS = -Wall -Wextra -Werror -Wno-unused-parameter -Iinclude -DSAFETY -lm
# Otherwise the stack gets smashed uhhhh

SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJS_NO_MAIN = $(filter-out $(BUILD_DIR)/main.o, $(OBJS))

TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS = $(TEST_SRCS:$(TEST_DIR)/%.c=$(BUILD_DIR)/test_%)

.PHONY: all clean test lint debug

all: $(BUILD_DIR) $(OBJS) $(TEST_BINS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

lint:
	clang-tidy $(SRCS) -- $(CFLAGS)

debug: CFLAGS += -g -O0
debug: clean all

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

clean:
	rm -rf $(BUILD_DIR)

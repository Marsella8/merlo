CC = gcc
CFLAGS = -Wall -Wextra -Werror -Wno-unused-parameter -Iinclude -DSAFETY
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS = $(TEST_SRCS:$(TEST_DIR)/%.c=$(BUILD_DIR)/test_%)

.PHONY: all clean test

all: $(BUILD_DIR) $(OBJS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_BINS)
	@for test in $(TEST_BINS); do \
		echo "Running $$test..."; \
		./$$test; \
		if [ $$? -ne 0 ]; then \
			echo "Test $$test failed!"; \
			exit 1; \
		fi; \
		echo "Test $$test passed"; \
	done

$(BUILD_DIR)/test_%: $(TEST_DIR)/%.c $(filter-out $(BUILD_DIR)/main.o, $(OBJS)) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(filter-out $(BUILD_DIR)/main.o, $(OBJS)) -o $@ -lm

clean:
	rm -rf $(BUILD_DIR)

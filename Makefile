CC = gcc
CFLAGS = -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-variable -Iinclude -lm -flto -march=native -DDEVICE_NAME=\"$(DEVICE_NAME)\"
DEVICE_NAME ?= HEAD

SRCS = $(wildcard src/*.c)
OBJS = $(SRCS:src/%.c=build/%.o)
LIB = $(filter-out build/main.o, $(OBJS))
TESTS = $(patsubst tests/%.c, build/test_%, $(wildcard tests/*.c))

.PHONY: all clean test debug release

all: CFLAGS += -O3
all: build/main $(TESTS)

debug: CFLAGS += -g -O0 -DSAFETY
debug: clean build/main $(TESTS)

release: CFLAGS += -O3
release: clean build/main $(TESTS)

test: all
	@for t in $(TESTS); do ./$$t && echo "passed" || echo "failed"; done

clean:
	rm -rf build

build:
	mkdir -p build

build/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/main: $(OBJS) | build
	$(CC) $(CFLAGS) $(OBJS) -o $@

build/test_%: tests/%.c $(LIB) | build
	$(CC) $(CFLAGS) $< $(LIB) -o $@

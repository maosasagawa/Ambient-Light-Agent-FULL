CC ?= gcc
CFLAGS ?= -std=c99 -Wall -Wextra -pedantic

.PHONY: all test clean

all: hw_self_test

hw_self_test: hw_gateway_sdk.c hw_gateway_sdk.h hw_gateway_sdk_self_test.c
	$(CC) $(CFLAGS) hw_gateway_sdk.c hw_gateway_sdk_self_test.c -o $@

test: hw_self_test
	./hw_self_test

clean:
	rm -f hw_self_test

# simple definitions for making libpi.a
#  - you should only ever have to modify SRC or STAFF_OBJS

# use the staff gpio implementation.
STAFF_OBJS  +=  ./staff-objs/gpio.o
SRC += src/gpio-int.c

# if you want to use your own pullup-puldown
# 1. put it in src/gpio-pud.c)
# 2. add it to SRC
# 3. comment out ours below:
STAFF_OBJS  +=  ./staff-objs/gpio-pud.o

STAFF_OBJS  +=  ./staff-objs/uart.o
STAFF_OBJS  +=  ./staff-objs/staff-sw-uart.o
STAFF_OBJS  +=  ./staff-objs/staff-kmalloc.o

# these are all the locations that get made into
# libpi.a
#
# you should not have to modify this variable.  
SRC :=  $(SRC)                              \
        $(wildcard ./libc/*.[Sc])           \
        $(wildcard ./staff-src/*.[Sc])      \
        staff-start.S

# hack to minimize git conflicts: we do various customizations
# in there; but probably would be clearer to inline it.
include ./manifest.mk

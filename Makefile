CFLAGS = -Wall -Wshadow -O3 -g -march=native
LDLIBS = -lm

all: test example

ffnn.o: ffnn.c network.pb-c.o

network.pb-c.o: extra/network.pb-c.c

test: ffnn.o

example: ffnn.o

clean:
	$(RM) *.o
	$(RM) test example extra/network.pb-c.o *.exe
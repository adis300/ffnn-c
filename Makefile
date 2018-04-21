CFLAGS = -Wall -Wshadow -O3 -g -march=native
LDLIBS = -lm

all: test example

test: test.o ffnn.o

example: example.o ffnn.o

clean:
	$(RM) *.o
	$(RM) test ffnn *.exe
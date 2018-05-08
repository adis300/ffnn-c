CFLAGS = -Wall -Wshadow -O3 -g -march=native
LDLIBS = -lm
LIBS=-libprotobuf-c

all: test example

ffnn.o: ffnn.c extra/network.pb-c.o

network.pb-c.o: extra/protobuf-c.o
protobuf-c.o: extra/protobuf-c.c

test: ffnn.o extra/network.pb-c.o extra/protobuf-c.o

example: ffnn.o extra/network.pb-c.o extra/protobuf-c.o

clean:
	$(RM) *.o
	$(RM) extra/*.o
	$(RM) test example *.exe

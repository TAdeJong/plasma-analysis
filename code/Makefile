#Compiler is nvcc
CC=nvcc
#Includeflags
IFLAGS=--include-path cudasamples

all: RK4test

RK4test: RK4test.cu
	$(CC) $(IFLAGS) RK4test.cu -o RK4test

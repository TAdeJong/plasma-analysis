#Compiler is nvcc
CC=nvcc
#Includeflags
IFLAGS=--include-path cudasamples -dc -c

all: main

main: main.o integration.o coordfunctions.o conversions.o
	$(CC) main.o coordfunctions.o integration.o -o main

%.o: %.cu
	$(CC) $(IFLAGS) -o $@ $<

clean: 
	rm *.o main

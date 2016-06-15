#Compiler is nvcc
CC=nvcc
#Includeflags
IFLAGS=--include-path cudasamples -dc -c

all: main

main: main.o integration.o coordfunctions.o
	$(CC) main.o coordfunctions.o integration.o -o main

integration.o: integration.cu
	$(CC) $(IFLAGS) integration.cu

main.o: main.cu
	$(CC) $(IFLAGS) main.cu

coordfunctions.o: coordfunctions.cu
	$(CC) $(IFLAGS) coordfunctions.cu

clean: 
	rm *.o main

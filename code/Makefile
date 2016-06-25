#Compiler is nvcc
CC=nvcc
#Includeflags
IFLAGS=--include-path cudasamples -dc -c

all: main

main: main.o integration.o coordfunctions.o conversions.o vtkio.o
	$(CC) main.o coordfunctions.o integration.o vtkio.o conversions.o -o main

%.o: %.cu
	$(CC) $(IFLAGS) -o $@ $<

clean: 
	rm *.o main

#Compiler is nvcc
CC=nvcc
#Includeflags
IFLAGS=-lineinfo --include-path cudasamples -dc -c
CCFLAGS= -arch compute_52

all: main

main: main.o integration.o coordfunctions.o conversions.o vtkio.o
	$(CC) $(CCFLAGS) main.o coordfunctions.o integration.o vtkio.o conversions.o -o main

%.o: %.cu
	$(CC) $(CCFLAGS) $(IFLAGS) -o $@ $<

clean: 
	rm *.o main

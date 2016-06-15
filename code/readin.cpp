#include <sys/mman.h>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <assert.h>
#include <fcntl.h>
#include <fstream>

using namespace std;

void readdata(float4* datapointer, int size, char* filename) {
	int filedescriptor = open(filename, O_RDONLY, 0);
	
	datapointer = (float4*)mmap(NULL, numberoffloats*sizeof(float4), PROT_READ, MAP_FILE | MAP_PRIVATE, filedescriptor, 0);
	assert(Data != MAP_FAILED);
	return;
}

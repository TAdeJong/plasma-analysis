#include <sys/mman.h>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <assert.h>
#include <fcntl.h>
#include <fstream>

using namespace std;

int main(int argc, char** argv) {
	
	ofstream floattest;
	floattest.open(argv[1]);
	float datapoint = 0.001;
	floattest << datapoint;
	floattest.close();

	int numberoffloats = 1;
	int filedescriptor = open(argv[1], O_RDONLY, 0);
	
	float* Data = (float*)mmap(NULL, numberoffloats*sizeof(float), PROT_READ, MAP_FILE | MAP_PRIVATE, filedescriptor, 0);
	assert(Data != MAP_FAILED);

	for (int i = 0; i < numberoffloats; i++) {
		cout << Data[i] << " ";
	}
	cout << endl;

	return 0;
}

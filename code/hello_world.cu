#include <stdio.h>

__global__ void nullkernel( void) {
}

int main(void) {
	nullkernel<<<1,1>>>();
	printf("Hello World!\n");
	return 0;
}

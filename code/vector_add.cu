#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main(void) {
	int *d_a, *d_b, *d_c;
	size_t size = 2*sizeof(int);
	int a[2] = {1,2};
	int b[2] = {1,2};
	int c[2] = {0,0};
	int test;
	cudaGetDeviceCount(&test);
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
	printf("c[0]%d\n",c[0]);
	printf("c[1]%d\n",c[1]);
	add<<<1,2>>>(d_a, d_b, d_c);
	a[1] = 6;
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&b, d_b, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(&a, d_a, size, cudaMemcpyDeviceToHost);
	printf("c[0]%d\n",c[0]);
	printf("c[1]%d\n",c[1]);
	printf("%p\n", &a);
	cudaFree(d_a); 
	cudaFree(d_b); 
	cudaFree(d_c); 
	return 0;
}

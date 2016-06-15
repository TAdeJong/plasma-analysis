#include <stdio.h>
#include <iostream>
#include <helper_cuda.h>
#include "constants.h"
#include "coordfunctions.h"
#include "integration.h"

//'Global' texture, declared as an external texture in integration.cu. Stores data on the device.
texture <float4, cudaTextureType3D, cudaReadModeElementType> dataTex;

/*	Generates a circular vectorfield around the origin for testing purposes.
	Note the order of the indices - the first index corresponds to the z coordinate,
	the middle to y and the last to x.
*/
void datagen (float4*** data) {
	//data[z][y][x]
	for (int i=0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				(data[i][j][k]).x = - (origin + spacing*j);
				(data[i][j][k]).y = (origin + spacing*k);
				(data[i][j][k]).z = 0;
				(data[i][j][k]).w = 0;
			}
		}
	}
}

int main(void) {
	//Allocate array on device
	cudaArray* dataArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
	cudaExtent extent = make_cudaExtent(N, N, N);
	checkCudaErrors(cudaMalloc3DArray(&dataArray, &channelDesc,extent));

	//Set linear interpolation mode
	dataTex.filterMode = cudaFilterModeLinear;

	//Allocate data array on host
	float4*** hostvfield; 
	hostvfield = (float4***) malloc(N*sizeof(float4**));
	hostvfield[0] = (float4**) malloc(N*N*sizeof(float4*));
	hostvfield[0][0] = (float4*) malloc(N*N*N*sizeof(float4));
	for (int i=1; i < N; i++) {
		hostvfield[i] = (hostvfield[0] + i*N);
	}
	for (int i=0; i < N; i++) {
		for (int j=0; j < N; j++) {
			hostvfield[i][j] = (hostvfield[0][0] + (i*N*N + j*N));
		}
	}

	//Generate data on host (used for testing)
	datagen(hostvfield);

	//Copy data to device
	cudaMemcpy3DParms copyParms = {0};
	copyParms.srcPtr = make_cudaPitchedPtr((void *)hostvfield[0][0], extent.width* sizeof(float4), extent.height, extent.depth);
	copyParms.dstArray = dataArray;
	copyParms.extent = extent;
	copyParms.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParms));

	//Copy our texture properties (linear interpolation, texture access) to data array on device
	checkCudaErrors(cudaBindTextureToArray(dataTex, dataArray, channelDesc));

	//Declare pointers to arrays with line data (output of integration), one each on device and host
	float4 *d_lines, *h_lines;

	//Set integration parameters (end time, number of steps, etc.)
	double time = 3.141592653*6.0;
	int steps = 100000;
	float dt = time/steps;

	dim3 gridsize(1,1);
	dim3 blocksize(1,1);
	int blocks = gridsize.x * gridsize.y;
	int cores = blocksize.x * blocksize.y;
	float4 startloc = {0,0,0,0};
	float4 xvec = {1,0,0,0};
	float4 yvec = {0,1,0,0};

	//Allocate space on device to store integration output
	checkCudaErrors(cudaMalloc(&d_lines,blocks*cores*steps*sizeof(float4)));

	//Allocate space on host to store integration output
	h_lines = (float4*) malloc(blocks*cores*steps*sizeof(float4));

	//Integrate the vector field
	RK4line<<<cores,blocks>>>(d_lines, dt, steps, startloc, xvec, yvec, gridsize);

	//Copy data from device to host
	checkCudaErrors(cudaMemcpy(h_lines, d_lines, blocks*cores*steps*sizeof(float4), cudaMemcpyDeviceToHost));

	//Print 100 samples from the line
	for(unsigned int i=0; i<100; i++) {
		std::cout << "x= " << h_lines[i*steps/100].x << "; y= "<< h_lines[i*steps/100].y << " "<< h_lines[i*steps/100].x*h_lines[i*steps/100].x+h_lines[i*steps/100].y*h_lines[i*steps/100].y << std::endl;
	}

	//Free host pointers
	free(hostvfield[0][0]);
	free(hostvfield[0]);
	free(hostvfield);
	
	return 0;
}


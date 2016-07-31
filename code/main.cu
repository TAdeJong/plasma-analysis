#include <stdio.h>
#include <helper_cuda.h>
#include <iostream>
#include "constants.cuh"
#include "coordfunctions.cuh"
#include "conversions.cuh"
#include "integration.cuh"
#include "vtkio.cuh"
#include "helper_math.h"

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



int main(int argc, char *argv[]) {
	if (argc == 1) {
		std::cout << "Please specify as an argument the path to the .vtk file to use"  << std::endl;
		return 1;
	}
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

	//Read data from file specified as argument
	float4 dataorigin = {0,0,0,0};
	vtkDataRead(hostvfield[0][0],argv[1], dataorigin);
	if(dataorigin.x != origin || dataorigin.y != origin || dataorigin.z != origin) {
		std::cout << "Warning: origin read from file not equal to origin from constants.h" << std::endl;
	}

	//Allocate array on device
	cudaArray* dataArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
	cudaExtent extent = make_cudaExtent(N, N, N);
	checkCudaErrors(cudaMalloc3DArray(&dataArray, &channelDesc,extent));

	//Set linear interpolation mode
	dataTex.filterMode = cudaFilterModeLinear;
	dataTex.addressMode[0] = cudaAddressModeBorder;
	dataTex.addressMode[1] = cudaAddressModeBorder;
	dataTex.addressMode[2] = cudaAddressModeBorder;


	//Copy data to device
	cudaMemcpy3DParms copyParms = {0};
	copyParms.srcPtr = make_cudaPitchedPtr((void *)hostvfield[0][0], extent.width*sizeof(float4), extent.height, extent.depth);
	copyParms.dstArray = dataArray;
	copyParms.extent = extent;
	copyParms.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParms));

	//Copy our texture properties (linear interpolation, texture access) to data array on device
	checkCudaErrors(cudaBindTextureToArray(dataTex, dataArray, channelDesc));

	//Declare pointers to arrays with line data (output of integration), one each on device and host
	float4 *d_lines, *h_lines;

	//Set integration parameters (end time, number of steps, etc.)
	const int blockSize = 1024;
	int steps = 8*blockSize;
	float dt = 1/8.0;

	dim3 gridsizeRK4(1,1);
	dim3 blocksizeRK4(8,8);
	int dataCount = gridsizeRK4.x*gridsizeRK4.y*blocksizeRK4.x*blocksizeRK4.y*steps;
	float4 startloc = make_float4(-1,0,0,0); //Location (in Smietcoords) to start the integration, to be varied
	float4 xvec = {1,0,0,0};
	float4 yvec = {0,1,0,0};

	//Allocate space on device to store integration output
	checkCudaErrors(cudaMalloc(&d_lines, dataCount*sizeof(float4)));

	//Allocate space on host to store integration output
	h_lines = (float4*) malloc(dataCount*sizeof(float4));

	//Integrate the vector field
	RK4line<<<gridsizeRK4,blocksizeRK4>>>(d_lines, dt, steps, startloc, xvec, yvec, gridsizeRK4);

	float4 *d_origins;
	checkCudaErrors(cudaMalloc(&d_origins, dataCount/(2*blockSize)*sizeof(float4)));

	//Add the coordinates of the streamlines coordinatewise (in order to calculate mean).
	reduceSum<<<dataCount/(2*blockSize),blockSize,blockSize*sizeof(float4)>>>(d_lines, d_origins);
	reduceSum<<<1,dataCount/(4*blockSize),dataCount/(4*blockSize)*sizeof(float4)>>>(d_origins, d_origins);

	//Copy data from device to host
	checkCudaErrors(cudaMemcpy(h_lines, d_lines, dataCount*sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&dataorigin, d_origins, sizeof(float4), cudaMemcpyDeviceToHost));

	//Converts the sum from above into an average
	dataorigin /= (float)steps*dataCount;

	//Display origin:
	std::cout << "Origin in Smiet: " << dataorigin.x << ", " << dataorigin.y << ", " << dataorigin.z << std::endl;
	float4 normal = {0,0,0,0};

	//Compute the normal to the plane through the torus. Reusing previously allocated d_origins
	reduceNormal<<<dataCount/(2*blockSize),blockSize,blockSize*sizeof(float4)>>>(d_lines, d_origins);
	reduceNormal<<<1,dataCount/(4*blockSize),dataCount/(4*blockSize)*sizeof(float4)>>>(d_origins, d_origins);

	//Copy normal from device to host
	checkCudaErrors(cudaMemcpy(&normal, d_origins, sizeof(float4), cudaMemcpyDeviceToHost));
	
	std::cout << "Normal: " << normal.x << ", " << normal.y << ", " << normal.z << std::endl;


	//Allocating the array to store the length data, both for host and device
	float *d_lengths, *h_lengths;
	checkCudaErrors(cudaMalloc(&d_lengths, dataCount*sizeof(float4)));
	h_lengths = (float*) malloc((dataCount/steps)*sizeof(float));

	//Compute the length of each line (locally)
	lineLength<<<dataCount/blockSize,blockSize>>>(d_lines, dt, d_lengths);

	//Add the length of the pieces of the lines to obtain line length
	//Stores the length of the i'th line in d_lengths[i]
	reduceSum<<<dataCount/steps,steps/2,(steps/2)*sizeof(float)>>>(d_lengths,d_lengths);

	//Copy lengths from device to host
	checkCudaErrors(cudaMemcpy(&h_lengths, d_lengths, (dataCount/steps)*sizeof(float4), cudaMemcpyDeviceToHost));
	
    //Write all the lines
    datawrite("../datadir/data.bin", dataCount, h_lines);
   
	//Free host pointers
	free(hostvfield[0][0]);
	free(hostvfield[0]);
	free(hostvfield);
	free(h_lines);
	free(h_lengths);
	cudaFree(d_origins);
	cudaFree(d_lines);
	cudaFree(d_lengths);
	cudaFreeArray(dataArray);
	return 0;
}


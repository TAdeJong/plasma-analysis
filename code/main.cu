#include <stdio.h>
#include <helper_cuda.h>
#include <iostream>
#include "constants.h"
#include "coordfunctions.h"
#include "conversions.h"
#include "integration.h"
#include "vtkio.h"
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
	double time = 1024;
	int steps = 1024;
	float dt = time/steps;

	dim3 gridsizeRK4(1,1);
	dim3 blocksizeRK4(8,8);
	int threadcountRK4 = gridsizeRK4.x*gridsizeRK4.y*blocksizeRK4.x*blocksizeRK4.y;
	float4 startloc = dataorigin; //Location (in Smietcoords) to start the integration, to be varied
	float4 xvec = {1,0,0,0};
	float4 yvec = {0,1,0,0};

	//Allocate space on device to store integration output
	checkCudaErrors(cudaMalloc(&d_lines, threadcountRK4*steps*sizeof(float4)));

	//Allocate space on host to store integration output
	h_lines = (float4*) malloc(threadcountRK4*steps*sizeof(float4));

	//Integrate the vector field
	RK4line<<<gridsizeRK4,blocksizeRK4>>>(d_lines, dt, steps, startloc, xvec, yvec, gridsizeRK4);
	float4 *d_origins;
	checkCudaErrors(cudaMalloc(&d_origins, threadcountRK4*sizeof(float4)));

	//Add the coordinates of the streamlines coordinatewise (in order to calculate mean.
	reduceSum<<<threadcountRK4,steps/2,steps/2*sizeof(float4)>>>(d_lines, d_origins);
	reduceSum<<<1,threadcountRK4/2,threadcountRK4*sizeof(float4)>>>(d_origins, d_origins);

	//Copy data from device to host
	checkCudaErrors(cudaMemcpy(h_lines, d_lines, threadcountRK4*steps*sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&dataorigin, d_origins, sizeof(float4), cudaMemcpyDeviceToHost));

	//Converts the sum from above into an average
	dataorigin /= (float)steps*threadcountRK4;

	//Display origin:
	std::cout << "Origin in Smiet: " << dataorigin.x << ", " << dataorigin.y << ", " << dataorigin.z << std::endl;
	float4 normal = {0,0,0,0};

	//Compute the normal to the plane through the torus. Reusing previously allocated d_origins
	reduceNormal<<<threadcountRK4,steps/2,steps/2*sizeof(float4)>>>(d_lines, d_origins);
	reduceNormal<<<1,threadcountRK4/2,threadcountRK4*sizeof(float4)>>>(d_origins, d_origins);
	checkCudaErrors(cudaMemcpy(&normal, d_origins, sizeof(float4), cudaMemcpyDeviceToHost));
	
	std::cout << "Normal: " << normal.x << ", " << normal.y << ", " << normal.z << std::endl;
/*	//Print 100 samples from the line
	int index = 0;
	for(unsigned int i=0; i<100; i++) {
		index = 2*steps + i*steps/100;
		std::cout << "x= " << h_lines[index].x << "; y= "<< h_lines[index].y << " "<< h_lines[index].x*h_lines[index].x+h_lines[index].y*h_lines[index].y << std::endl;
	}*/
    
//    datawrite("../datadir/test.bin", steps, h_lines);
   
//Free host pointers
	free(hostvfield[0][0]);
	free(hostvfield[0]);
	free(hostvfield);
	free(h_lines);
	cudaFree(d_origins);
	cudaFree(d_lines);
	cudaFreeArray(dataArray);
	return 0;
}


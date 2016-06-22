#include <stdio.h>
#include <helper_cuda.h>
#include <iostream>
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

int dataread (float4* data, const char* filename, float4 &origin){
	unsigned int i, j, k, datasize;
	unsigned int n_x =0, n_y=0, n_z=0;
	char kind[20];
	char name[20];
	char type[20];
	char rstr[80];
	float4 rspacing;
	FILE *dfp;

	dfp= fopen(filename, "r");
	for(unsigned int i=0; i<4; ++i) {
		fgets(rstr, 80, dfp);
		std::cout << rstr;
	}
	fscanf(dfp, "%s %u %u %u", rstr, &n_x, &n_y, &n_z);
	if(!( n_x == 256 && n_y == 256 && n_z ==256)) {
		std::cout<<"Warning: incorrect " << rstr << " read: expected 256, got: " << n_z << std::endl;
	}
	fscanf(dfp, "%s %f %f %f", rstr, &origin.x, &origin.y, &origin.z);
	fscanf(dfp, "%s %f %f %f", rstr, &rspacing.x, &rspacing.y, &rspacing.z);
	if(! (rspacing.x == rspacing.y && rspacing.y == rspacing.z)) {
		std::cout << "Warning: (unsupported) anisotrope spacing read!" << std::endl;
	}
	fscanf(dfp, "%s %u", rstr, &datasize);
	if(datasize != n_x*n_y*n_z) {
		std::cout<<"Error: " << rstr << "is not equal to n_x*n_y*n_z" << std::endl;
		return 1;
	}
	fscanf(dfp, "%s %s %s", kind, name, type);
	if(kind != "VECTORS" || name != "bfield" || type != "float") {
		std::cout << "Error: Incorrect kind, name or type" << std::endl;
		return 1;
	}
	for(unsigned int i=0; i<datasize; ++i) {
		float datapoint[3] = {0,0,0};
		fread(datapoint, sizeof(float), 3, dfp);
		data[i] = make_float4(datapoint[0],datapoint[1],datapoint[2],0);
	}
	std::cout << "Data read in was succesfull!" << std::endl;
	return 0;
}


void datawrite (const char* location, int steps, float4* h_lines){ 
    //write the first streamline to a file. Remember this is 32 bits when reading!
    FILE *fp;
    fp = fopen(location, "w");
    for (unsigned int i = 0; i<steps; i++){   //write only the first streamline
        fwrite(&h_lines[i], sizeof(float4), 1, fp);
    }
    fclose(fp);
    std::cout<<"streamline written!"<<std::endl;
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

	dim3 gridsizeRK4(1,1);
	dim3 blocksizeRK4(8,8);
	int threadcountRK4 = gridsizeRK4.x*gridsizeRK4.y*blocksizeRK4.x*blocksizeRK4.y;
	float4 startloc = {1,0,0,0};
	float4 xvec = {1,0,0,0};
	float4 yvec = {0,1,0,0};

	//Allocate space on device to store integration output
	checkCudaErrors(cudaMalloc(&d_lines, threadcountRK4*steps*sizeof(float4)));

	//Allocate space on host to store integration output
	h_lines = (float4*) malloc(threadcountRK4*steps*sizeof(float4));

	//Integrate the vector field
	RK4line<<<gridsizeRK4,blocksizeRK4>>>(d_lines, dt, steps, startloc, xvec, yvec, gridsizeRK4);

	//Copy data from device to host
	checkCudaErrors(cudaMemcpy(h_lines, d_lines, threadcountRK4*steps*sizeof(float4), cudaMemcpyDeviceToHost));

	//Print 100 samples from the line
	int index = 0;
	for(unsigned int i=0; i<100; i++) {
		index = 2*steps + i*steps/100;
		std::cout << "x= " << h_lines[index].x << "; y= "<< h_lines[index].y << " "<< h_lines[index].x*h_lines[index].x+h_lines[index].y*h_lines[index].y << std::endl;
	}
    
//    datawrite("../datadir/test.bin", steps, h_lines);
    dataread(hostvfield[0][0],"test.txt", startloc);
            
    //Free host pointers
	free(hostvfield[0][0]);
	free(hostvfield[0]);
	free(hostvfield);
        
	
	return 0;
}


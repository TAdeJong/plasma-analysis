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

void allocarray (float4*** &hostvfield) {
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
}

void dataprint (float* data, dim3 Size) {
	for(unsigned int i=0; i< Size.x; i++) {
		for(unsigned int j=0; j< Size.y; j++) {
			std::cout << data[Size.y*i+j] << " , ";
		}
		std::cout << std::endl;
	}
}

void dataprint (float4* data, dim3 Size) {
	for(unsigned int i=0; i< Size.x; i++) {
		for(unsigned int j=0; j< Size.y; j++) {
			std::cout << data[Size.y*i+j].x << ", "<< data[Size.y*i+j].y << ", "<< data[Size.y*i+j].z << std::endl;
		}
		std::cout << std::endl;
	}
}

int main(int argc, char *argv[]) {

	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	std::cout<<"using "<<properties.multiProcessorCount<<" multiprocessors"<<std::endl;
	std::cout<<"max threads per processor: "<<properties.maxThreadsPerMultiProcessor<<std::endl;

	//Check if the input is sensible
	std::string name;
	if (argc == 1) {
		std::cout << "Please specify as an argument the path to the .vtk file to use"  << std::endl;
		return 1;
	} else {
		name = argv[1];
	}
	if (name.rfind(".vtk") == std::string::npos) {
		name.append(".vtk");
	}
	//Allocate data array on host
	float4*** hostvfield;
	allocarray(hostvfield);

	//Read data from file specified as argument
	float4 dataorigin = {0,0,0,0};
	vtkDataRead(hostvfield[0][0], name.c_str(), dataorigin);
	if(dataorigin.x != origin || dataorigin.y != origin || dataorigin.z != origin) {
		std::cout << "Warning: origin read from file not equal to origin from constants.h" << std::endl;
	}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

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

	//Copy data (originally from the vtk) to device
	cudaMemcpy3DParms copyParms = {0};
	copyParms.srcPtr = make_cudaPitchedPtr((void *)hostvfield[0][0],
		   	extent.width*sizeof(float4),
		   	extent.height,
		   	extent.depth
			);
	copyParms.dstArray = dataArray;
	copyParms.extent = extent;
	copyParms.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParms));

	//Copy our texture properties (linear interpolation, texture access) to data array on device
	checkCudaErrors(cudaBindTextureToArray(dataTex, dataArray, channelDesc));

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

	//Declare pointers to arrays with line data (output of integration), one each on device and host
	float4 *d_lines, *h_lines;

	//Set integration parameters (end time, number of steps, etc.)
	const int blockSize = 1024;
	unsigned int steps = 32*blockSize;
	float dt = 1/2.0;

	dim3 BIGgridSize(32,32);
	dim3 gridSizeRK4(4,4);
	dim3 blockSizeRK4(16,16); //gridSizeRK4*blockSizeRK4*steps should not exceed 2^26, to fit on 4GM VRAM

	int dataCount = gridSizeRK4.x*gridSizeRK4.y*blockSizeRK4.x*blockSizeRK4.y*steps;
	int BIGnroflines = BIGgridSize.x*BIGgridSize.y*blockSizeRK4.x*blockSizeRK4.y;

//	float4 BIGstartloc = make_float4(0.75,0,-0.45,0); //Location (in Smietcoords) to start the 
//	integration, to be varied
	float4 BIGstartloc = make_float4(0,0,-1.25,0); //Location (in Smietcoords) to start the integration, to be varied
	float4 BIGxvec = {1.75,0,0,0};
	float4 BIGyvec = {0,0,1.75,0};

	//Allocate host arrays for the winding numbers,
	float* h_windingdata;
	cudaError_t status = cudaMallocHost((void**)&h_windingdata,BIGnroflines*sizeof(float));
	if (status != cudaSuccess)
		printf("Error allocating pinned host memory.\n");
	
	//Allocate space on device & host to store integration output
	checkCudaErrors(cudaMalloc(&d_lines, dataCount*sizeof(float4)));
	h_lines = (float4*) malloc(dataCount*sizeof(float4));

	//Allocate space to store origin data
	float4 *d_origins;
//	       *h_origins;
	checkCudaErrors(cudaMalloc(&d_origins, dataCount/(2*blockSize)*sizeof(float4)));
//	h_origins = (float4*) malloc(BIGnroflines * sizeof(float4));

	//Allocating the array to store the length data, both for host and device
	float *d_lengths, *h_lengths;
	checkCudaErrors(cudaMalloc(&d_lengths, dataCount*sizeof(float)));
	status = cudaMallocHost((void**)&h_lengths, BIGnroflines*sizeof(float));
	if (status != cudaSuccess)
		printf("Error allocating pinned host memory.\n");
	//Allocating Origins

	float4 *d_normals;
	checkCudaErrors(cudaMalloc(&d_normals, dataCount*sizeof(float4)));

	//Allocating the array to store the radius data, both for host and device
	float *d_radii;
//	      *h_radii;
	checkCudaErrors(cudaMalloc(&d_radii, dataCount*sizeof(float)));
//	h_radii = (float*) malloc(BIGnroflines*sizeof(float));

	//Allocating arrays to store the alpha and beta data to compute winding numbers.
	float *d_alpha, *d_beta;
	checkCudaErrors(cudaMalloc(&d_alpha, dataCount*sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_beta, dataCount*sizeof(float)));



	//Set up streams for independent execution
	cudaStream_t RK4, windings, windings2, lengths;
	status = cudaStreamCreate(&RK4);
	status = cudaStreamCreate(&windings);
	status = cudaStreamCreate(&windings2);
	status = cudaStreamCreate(&lengths);


	//Start main loop.
	for (int yindex = 0; yindex < BIGgridSize.y; yindex += gridSizeRK4.y) {
		for (int xindex = 0; xindex < BIGgridSize.x; xindex += gridSizeRK4.x) {

			std::cout << "x" << std::flush;
			float4 startloc = BIGstartloc + ((float)xindex/BIGgridSize.x) * BIGxvec + ((float)yindex/BIGgridSize.y) * BIGyvec;
			float4 xvec = BIGxvec * ((float)gridSizeRK4.x/BIGgridSize.x);
			float4 yvec = BIGyvec * ((float)gridSizeRK4.y/BIGgridSize.y);

			int globaloffset = yindex*blockSizeRK4.y*BIGgridSize.x*blockSizeRK4.x+xindex*blockSizeRK4.x;

			int hsize = gridSizeRK4.x*blockSizeRK4.x;
			int vsize = gridSizeRK4.y*blockSizeRK4.y;

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

			cudaDeviceSynchronize();	
			cudaStreamSynchronize(windings);
			cudaStreamSynchronize(lengths);
			//Integrate the vector field
			RK4line<<<gridSizeRK4,blockSizeRK4,0,RK4>>>(d_lines, dt, steps, startloc, xvec, yvec);

			//Copy data from device to host
//			checkCudaErrors(cudaMemcpy(h_lines, d_lines, dataCount*sizeof(float4), cudaMemcpyDeviceToHost));

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

			//cudaDeviceSynchronize();	
			cudaStreamSynchronize(RK4);//Wait for d_lines to be filled
	
			//Compute the length of each line (locally)
			lineLength<<<dataCount/blockSize,blockSize,0,lengths>>>(d_lines, dt, d_lengths);

			//Add the length of the pieces of the lines to obtain line length
			//Stores the length of the i'th line in d_lengths[i]
			reduceSum<float><<<dataCount/(2*blockSize),blockSize,blockSize*sizeof(float),lengths>>>
				(d_lengths,d_lengths);
			reduceSum<float><<<dataCount/steps,steps/(4*blockSize),steps/(4*blockSize)*sizeof(float),lengths>>>
				(d_lengths,d_lengths);

			//Copy lengths from device to host
			//Note: can be even more asynchronous and should copy to different parts of h_lengths.
/*			checkCudaErrors(cudaMemcpyAsync(h_lengths, 
						d_lengths, 
						(dataCount/steps)*sizeof(float), 
						cudaMemcpyDeviceToHost,
						lengths));*/
			checkCudaErrors(cudaMemcpy2DAsync(
						&(h_lengths[globaloffset]),
						BIGgridSize.x*blockSizeRK4.x*sizeof(float),
					   	d_lengths,
						hsize*sizeof(float), 
						hsize*sizeof(float), 
						vsize, 
						cudaMemcpyDeviceToHost,
						lengths
						));

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
			
			cudaStreamSynchronize(lengths);
			//Make sure data from previous iteration is copied away to host
			cudaStreamSynchronize(windings2);
			//Add the coordinates of the streamlines coordinatewise (in order to calculate mean).
			reduceSum<float4><<<dataCount/(2*blockSize),blockSize,blockSize*sizeof(float4),windings>>>
				(d_lines, d_origins);
			reduceSum<float4><<<dataCount/steps,steps/(4*blockSize),steps/(4*blockSize)*sizeof(float4),windings>>>
				(d_origins, d_origins);
			divide<<<dataCount/(steps*blockSize),blockSize,0,windings>>>
				(d_origins,(float)steps, d_origins);//not size-scalable!!!


		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//


		//Calculate the normals.
			normal<<<dataCount/blockSize,blockSize,0,windings>>>
				(d_lines, d_normals, d_origins, steps);
			reduceSum<float4><<<dataCount/(2*blockSize),blockSize,blockSize*sizeof(float4),windings>>>
				(d_normals, d_normals);
			reduceSum<float4><<<dataCount/steps,steps/(4*blockSize),steps/(4*blockSize)*sizeof(float4),windings>>>
				(d_normals, d_normals);
			divide<<<dataCount/(steps*blockSize),blockSize,0,windings>>>
				(d_normals,(float)steps, d_normals);//not size-scalable!!!


//			cudaDeviceSynchronize();	
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//



			//Compute the distance from the origin in the xy plane of each point
			rxy<<<dataCount/blockSize,blockSize,0,windings>>>
				(d_lines, d_radii, (float)steps, d_origins, d_normals, steps);

			//Average these distances to find the torus radius
			reduceSum<float><<<dataCount/(2*blockSize),blockSize,blockSize*sizeof(float),windings>>>
				(d_radii,d_radii);
			reduceSum<float><<<dataCount/steps,steps/(4*blockSize),steps/(4*blockSize)*sizeof(float),windings>>>
				(d_radii,d_radii);



//			cudaDeviceSynchronize();	
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
			winding<<<dataCount/blockSize,blockSize,0,windings>>>(d_lines, d_alpha, d_beta, 
d_origins, d_radii, d_normals, steps);

			//Adding the steps Deltaalpha and Deltabeta to find overall windings
			//This code is dependent on completion of winding, but independent on d_lines
			cudaStreamSynchronize(windings);//Wait for d_lines to be filled
			reduceSum<float>
				<<<dataCount/(2*blockSize),blockSize,blockSize*sizeof(float),windings2>>>
				(d_alpha, d_alpha);
			reduceSum<float>
				<<<dataCount/steps,steps/(4*blockSize),steps/(4*blockSize)*sizeof(float),windings2>>>
				(d_alpha, d_alpha);

			reduceSum<float>
				<<<dataCount/(2*blockSize),blockSize,blockSize*sizeof(float),windings2>>>
				(d_beta, d_beta);
			reduceSum<float>
				<<<dataCount/steps,steps/(4*blockSize),steps/(4*blockSize)*sizeof(float),windings2>>>
				(d_beta, d_beta);

			//Dividing these windings to compute the winding numbers and store them in d_alpha
			divide<<<dataCount/(steps*blockSize),blockSize,0,windings2>>>
				(d_beta, d_alpha, d_alpha);//Not Scalable!!!

//			cudaDeviceSynchronize();	
			checkCudaErrors(cudaMemcpy2DAsync(
						&(h_windingdata[globaloffset]),
						BIGgridSize.x*blockSizeRK4.x*sizeof(float),
					   	d_alpha,
						hsize*sizeof(float), 
						hsize*sizeof(float), 
						vsize, 
						cudaMemcpyDeviceToHost,
						windings2
						));
		}
		std::cout << std::endl;
	}

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
		   
			//Free host pointers
//			free(h_origins);
			free(h_lines);

			//Free the remaining device pointers
			cudaFree(d_alpha);
			cudaFree(d_beta);
			cudaFree(d_radii);
			cudaFree(d_lengths);
			cudaFree(d_origins);
			cudaFree(d_lines);

//Print some data to screen
	//	dim3 printtest(16,16);
	//	dataprint(h_windingdata,printtest);
//Write some data
//	float4write("../datadir/linedata.bin", BIGdataCount, h_lines);
	name = name.substr(name.rfind("/")+1,name.rfind(".")-name.rfind("/")-1);
	const std::string prefix = "../datadir/";
	std::string suffix = "_windings.bin";
	std::string path = prefix+name+suffix;
	floatwrite(path.c_str(), BIGnroflines, h_windingdata);
	suffix = "_lengths.bin";
	path = prefix+name+suffix;
	floatwrite(path.c_str(), BIGnroflines, h_lengths);

	status = cudaStreamDestroy(RK4);
	status = cudaStreamDestroy(windings);
	status = cudaStreamDestroy(lengths);


	free(hostvfield[0][0]);
	free(hostvfield[0]);
	free(hostvfield);

	cudaFreeHost(h_windingdata);
	cudaFreeArray(dataArray);
	cudaFreeHost(h_lengths);
	return 0;
}


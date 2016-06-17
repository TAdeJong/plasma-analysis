#include "coordfunctions.h"
#include "constants.h"
#include "conversions.h"

/*	Uses parallel computing to determine the origin (middle) of each of the field lines
	computed with RK4 and stored in lineoutput earlier.
	Warning: only works when the total number of threads used to call this function
	is a multiple of numberoflines, and their ratio is a divisor of the blocksize
	(so each RK4 line will be processed within a single block)
*/
__device__ float4 origin(float4* lineoutput, int steps, dim3 gridsize, int numberoflines, float4* communication) {
	float4 origin = {0,0,0,0};
	dim3 index2D(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y*blockDim.y);
	int index = index2D.y*(gridsize.x*blockDim.x) + index2D.x;
	int threadsperline = blockDim.x*gridsize.x*Blockdim.y*gridsize.y/numberoflines;
	int rangeperthread = steps/threadsperline;
	if (blockDim.x*blockDim.y % threadsperline == 0) { //yay
		for (int i=0; i < rangeperthread; i++) {
			origin += lineoutput[rangeperthread*index+i];
		}
		origin *= (1.0/localrange);
		communication[index] = origin;
		__syncthreads();
		int threadindex = (index - (index % threadsperline));
		origin = {0,0,0,0};
		for (int i=0; i < threadsperline; i++) {
			origin += communication[threadindex + i];
		}
		origin *= (1.0/threadsperline);
	} else { //noooo
		
	}
	return origin;
}

__device__ float4 normal(float4* lineoutput, int steps, dim3 gridsize, int numberoflines, float4* communication, float4 origin) {
	float4 normal = {0,0,0,0};
	dim3 index2D(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y*blockDim.y);
	int index = index2D.y*(gridsize.x*blockDim.x) + index2D.x;
	int threadsperline = blockDim.x*gridsize.x*Blockdim.y*gridsize.y/numberoflines;
	int rangeperthread = steps/threadsperline;
	if (blockDim.x*blockDim.y % threadsperline == 0) { //yay
		for (int i=0; i < rangeperthread; i++) {
			normal += cross(make_float3(lineoutput[rangeperthread*index+i]),make_float3(lineoutput[rangeperthread*index+i+1] - lineoutput[rangeperthread*index+i]));
		}
		communication[index] = normal;
		__syncthreads();
		int threadindex = (index - (index % threadsperline));
		normal = {0,0,0,0};
		for (int i=0; i < threadsperline; i++) {
			normal += communication[threadindex + i];
		}
		normal += cross(make_float3(origin),make_float3(lineoutput[(threadindex+1)*steps-1]-lineoutput[threadindex*steps]));
		normal *= (1.0/steps);
	} else { //noooo
		
	}
	return normal;
}

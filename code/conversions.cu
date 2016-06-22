#include "coordfunctions.h"
#include "constants.h"
#include "conversions.h"
#include "helper_math.h"

extern texture <float4, cudaTextureType3D, cudaReadModeElementType> dataTex;

/*	Uses parallel computing to determine the origin (middle) of each of the field lines
	computed with RK4 and stored in lineoutput earlier.
	Warning: only works when the total number of threads used to call this function
	is a multiple of numberoflines, and their ratio is a divisor of the blocksize
	(so each RK4 line will be processed within a single block)
*/
__device__ float4 calcorigin(float4* lineoutput, int steps, dim3 gridsize, int numberoflines, float4* communication) {
	float4 origin = make_float4(0.0);
	dim3 index2D(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y*blockDim.y);
	int index = index2D.y*(gridsize.x*blockDim.x) + index2D.x;
	int threadsperline = blockDim.x*gridsize.x*blockDim.y*gridsize.y/numberoflines;
	int rangeperthread = steps/threadsperline;
	if (blockDim.x*blockDim.y % threadsperline == 0) { //yay
		for (unsigned int i=0; i < rangeperthread; i++) {
			origin += lineoutput[rangeperthread*index+i];
		}
		communication[index] = origin;
		__syncthreads();
		int threadindex = (index - (index % threadsperline));
		origin = make_float4(0.0);
		for (unsigned int i=0; i < threadsperline; i++) {
			origin += communication[threadindex + i];
		}
		origin *= (1.0/steps);
	} else { //noooo
		
	}
	return origin;
}

__device__ void reducesum(float4* g_linedata, float4* g_sumdata) {
	extern __shared__ float4 sdata[];

	//load data from global data to shared mem and perform first reduction
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	sdata[i] = g_linedata[i]+g_linedata[i+blockDim.x];
	__syncthreads();

	//do the reductions
	for( unsigned int s=blockDim.x/2; s>32; s>>=1) {//32 = warpsize
		if(tid < s) {
			sdata[tid] += sdata[tid+s];
		}

		__syncthreads();
	}
	if(tid<32) {// Warp's zijn SIMD gesynchroniseerd
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}
	//write result to global
	if(tid == 0) g_sumdata[blockIdx.x] = sdata[0];
}




__device__ float4 calcnormal(float4* lineoutput, int steps, dim3 gridsize, int numberoflines, float4* communication, float4 origin) {
	float4 normal = make_float4(0.0);
	dim3 index2D(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y*blockDim.y);
	int index = index2D.y*(gridsize.x*blockDim.x) + index2D.x;
	int threadsperline = blockDim.x*gridsize.x*blockDim.y*gridsize.y/numberoflines;
	int rangeperthread = steps/threadsperline;
	if (blockDim.x*blockDim.y % threadsperline == 0) { //yay
		for (int i=0; i < rangeperthread; i++) {
			normal += make_float4(cross(make_float3(lineoutput[rangeperthread*index+i]),make_float3(lineoutput[rangeperthread*index+i+1] - lineoutput[rangeperthread*index+i])));
		}
		communication[index] = normal;
		__syncthreads();
		int threadindex = (index - (index % threadsperline));
		normal = make_float4(0.0);
		for (int i=0; i < threadsperline; i++) {
			normal += communication[threadindex + i];
		}
		normal += make_float4(cross(make_float3(origin),make_float3(lineoutput[(threadindex+1)*steps-1]-lineoutput[threadindex*steps])));
		normal *= (1.0/steps);
	} else { //noooo
		
	}
	return normal;
}
//Give a third parameter to your kernellaunch for the size of sdata
__device__ void reducenormal(float4* g_linedata, float4* g_normaldata) {//equivalent to doing the texture-fetch and cross product and applying reducesum
	extern __shared__ float4 sdata[];

	//load data from global data&texture to shared mem and perform cross product
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[i] = make_float4(cross(make_float3(g_linedata[i]), make_float3(tex3D(dataTex, Smiet2Tex(g_linedata[i])))));
	__syncthreads();

	//do the reductions
	for( unsigned int s=blockDim.x; s>32; s>>=1) {//32 = warpsize
		if(tid < s) {
			sdata[tid] += sdata[tid+s];
		}

		__syncthreads();
	}
	if(tid<32) {// Warp's zijn SIMD gesynchroniseerd
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}
	//write result to global
	if(tid == 0) g_normaldata[blockIdx.x] = sdata[0];
}

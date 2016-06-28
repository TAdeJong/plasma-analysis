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

__device__ int signdiff(float a, float b) {
	return (a < 0 && b >= 0) || (a>0 && b <=0);
}

//find the number of x=0 transitions in g_linedata, storing the result in g_sumdata. Only works for powers of 2 datasets and needs a minimum of sdata of 64*sizeof(float) (!)
__global__ void reducePC(float4* g_linedata, float4* g_PCdata) {
	extern __shared__ int sdata[];
	//load data from global data&texture to shared mem and perform cross product
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = signdiff(g_linedata[i].x,g_linedata[i+1].x);
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
	if(tid == 0) g_normaldata[blockIdx.x] = sdata[0];
}

//Sum all elements in g_linedata, storing the result in g_sumdata. Only works for powers of 2 datasets and needs a minimum of sdata of 64*sizeof(float4) (!)
__global__ void reduceSum(float4* g_linedata, float4* g_sumdata) {
	extern __shared__ float4 sdata[];

	//load data from global data to shared mem and perform first reduction
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	sdata[tid] = g_linedata[i]+g_linedata[i+blockDim.x];
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



//Give a third parameter to your kernellaunch for the size of sdata
__global__ void reduceNormal(float4* g_linedata, float4* g_normaldata) {//equivalent to doing the texture-fetch and cross product and applying reducesum
	extern __shared__ float4 sdata[];

	//load data from global data&texture to shared mem and perform cross product
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = make_float4(cross(make_float3(g_linedata[i]), make_float3(tex3D(dataTex, Smiet2Tex(g_linedata[i])))));
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
	if(tid == 0) g_normaldata[blockIdx.x] = sdata[0];
}

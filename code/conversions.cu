#include "coordfunctions.cuh"
#include "constants.cuh"
#include "conversions.cuh"
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
__global__ void reducePC(float4* g_linedata, int* g_PCdata) {
	extern __shared__ int idata[];
	//load data from global data&texture to shared mem and perform cross product
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	idata[tid] = signdiff(g_linedata[i].x,g_linedata[i+1].x);
	__syncthreads();

	//do the reductions
	for( unsigned int s=blockDim.x/2; s>32; s>>=1) {//32 = warpsize
		if(tid < s) {
			idata[tid] += idata[tid+s];
		}

		__syncthreads();
	}
	if(tid<32) {// Warp's zijn SIMD gesynchroniseerd
		idata[tid] += idata[tid + 32];
		idata[tid] += idata[tid + 16];
		idata[tid] += idata[tid + 8];
		idata[tid] += idata[tid + 4];
		idata[tid] += idata[tid + 2];
		idata[tid] += idata[tid + 1];
	}
	//write result to global
	if(tid == 0) g_PCdata[blockIdx.x] = idata[0];
}

//Sum all elements in g_linedata, storing the result in g_sumdata. Only works for powers of 2 datasets
__global__ void reduceSum(float4* g_linedata, float4* g_sumdata) {
	extern __shared__ float4 sdata[];

	//load data from global data to shared mem and perform first reduction
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	sdata[tid] = g_linedata[i]+g_linedata[i+blockDim.x];
	__syncthreads();

	//do the reductions
	unsigned int s=blockDim.x/2;
	for( ; s>32; s>>=1) {//32 = warpsize
		if(tid < s) {
			sdata[tid] += sdata[tid+s];
		}

		__syncthreads();
	}
	if(tid < s) {
		for( ; s>0; s>>=1) {// Warp's zijn SIMD gesynchroniseerd Loop-unroll would require a Template-use
			sdata[tid] += sdata[tid+s];
		}
	}

	//write result to global
	if(tid == 0) g_sumdata[blockIdx.x] = sdata[0];
}

/*	Sums all floats in g_linedata, storing the result in g_sumdata.
	Only works for powers of 2 datasets and needs a minimum of sdata of 64*sizeof(float4) (!)
	Identical to reduceSum for float4's, needless copying can be fixed with templates
	but requires clever inclusions of code throughout files. Maybe to be added later.
*/
__global__ void reduceSum(float* g_linedata, float* g_sumdata) {
	extern __shared__ float shdata[];

	//load data from global data to shared mem and perform first reduction
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	shdata[tid] = g_linedata[i]+g_linedata[i+blockDim.x];
	__syncthreads();

	//do the reductions
	unsigned int s = blockDim.x/2;
	for( ; s>32; s>>=1) {//32 = warpsize
		if(tid < s) {
			shdata[tid] += shdata[tid+s];
		}

		__syncthreads();
	}
	if(tid < s) {
		for( ; s>0; s>>=1) {// Warp's zijn SIMD gesynchroniseerd Loop-unroll would require a Template-use
			shdata[tid] += shdata[tid+s];
		}
	}

	//write result to global
	if(tid == 0) g_sumdata[blockIdx.x] = shdata[0];
}


/*	Warning: absolutely useless!
	Mathematics is not correct, does not give normal to plane of torus!!
		DO NOT USE
	Give a third parameter to your kernellaunch for the size of sdata
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
}*/

__global__ void winding(float4* g_linedata, float4* g_windingdata, float4* origin, float* g_rdata, unsigned int steps) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int modifier = min(i%steps,1);
	float r_t = g_rdata[i/steps];
	float4 locCord = Cart2Tor(ShiftCoord(g_linedata[i], origin[i/steps]), r_t);
	locCord -= Cart2Tor(ShiftCoord(g_linedata[i-modifier], origin[i/steps]), r_t);
	//lelijk en langzaam, maar mijn bit-wise magic is niet genoeg om dit netjes te doen
	if(locCord.y > PI) {
		locCord.y -= 2*PI;
	} else if (locCord.y< -1*PI) {
		locCord.y += 2*PI;
	}
	if(locCord.z > PI) {
		locCord.z -= 2*PI;
	} else if (locCord.z < -1*PI) {
		locCord.z += 2*PI;
	}
	g_windingdata[i] =  locCord;
}

__global__ void divide(float* enumerator, float* denominator, float* output) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	output[i] = enumerator[i]/denominator[i];
}

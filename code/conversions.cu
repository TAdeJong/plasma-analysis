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
	for( unsigned int s=blockDim.x/2; s>32; s>>=1) {//32 = warpsize
		if(tid < s) {
			shdata[tid] += shdata[tid+s];
		}

		__syncthreads();
	}
	if(tid<32) {// Warp's zijn SIMD gesynchroniseerd
		shdata[tid] += shdata[tid + 32];
		shdata[tid] += shdata[tid + 16];
		shdata[tid] += shdata[tid + 8];
		shdata[tid] += shdata[tid + 4];
		shdata[tid] += shdata[tid + 2];
		shdata[tid] += shdata[tid + 1];
	}
	//write result to global
	if(tid == 0) g_sumdata[blockIdx.x] = shdata[0];
}



/*	Warning: absolutely useless!
	Mathematics is not correct, does not give normal to plane of torus!!
		DO NOT USE
	Give a third parameter to your kernellaunch for the size of sdata
*/
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

__device__ float Lengthstep(float4 loc, double dt) {
	float3 loc3dTex = Smiet2Tex(loc);
	float4 k1 = tex3D(dataTex, loc3dTex);
	float4 k2 = tex3D(dataTex, loc3dTex+(dt*0.5/spacing)*make_float3(k1));
	float4 k3 = tex3D(dataTex, loc3dTex+(dt*0.5/spacing)*make_float3(k2));
	float4 k4 = tex3D(dataTex, loc3dTex+(dt/spacing)*make_float3(k3));
	float l1 = length(k1);
	float l2 = length(k2);
	float l3 = length(k3);
	float l4 = length(k4);
	return dt/6.0*(l1 + 2.0*(l2 + l3) + l4);
}

__global__ void lineLength(float4* g_linedata, double dt, float* g_lengthoutput) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	g_lengthoutput[index] = Lengthstep(g_linedata[index],dt);
}

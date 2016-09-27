/* 	Contains the functions relating to transforming the Cartesian coordinate system into
	donut coordinates, including finding the middle of a field line
*/

#ifndef _CONVERSIONS_H_
#define _CONVERSIONS_H_
__device__ int signdiff(float a, float b);

__global__ void reducePC(float4* g_linedata, float4* g_PCdata);

__global__ void normal(float4* g_linedata, float4* g_normaldata, float4* g_origin, unsigned int steps);

//Sum all elements in g_linedata, storing the result in g_sumdata. Only works for powers of 2 datasets
template <typename T> 
__global__ void reduceSum(T* g_linedata, T* g_sumdata) {
	extern __shared__ float4 extdata[];
	T* sdata = (T*) extdata;

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

__global__ void winding(float4* g_linedata ,float* g_alpha,float* g_beta, float4* origin, float* r_t, 
float4* d_normals, unsigned int steps);

__global__ void divide(float* enumerator, float* denominator, float* output);
__global__ void divide(float4* enumerator, float denominator, float4* output);

//__global__ void reduceNormal( float4* g_linedata, float4* g_normaldata);

#endif

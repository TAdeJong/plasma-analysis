#include "coordfunctions.cuh"
#include "constants.cuh"
#include "integration.cuh"
#include "helper_math.h"
extern texture <float4, cudaTextureType3D, cudaReadModeElementType> dataTex;

//Performs a single RK4 step. Both input and output are in Smietcoordinates
__device__ float4 RK4step(float4 loc, double dt) {
	float3 loc3dTex = Smiet2Tex(loc);
	float4 k1 = tex3D(dataTex, loc3dTex);
	float4 k2 = tex3D(dataTex, loc3dTex+(dt*0.5/spacing)*make_float3(k1));
	float4 k3 = tex3D(dataTex, loc3dTex+(dt*0.5/spacing)*make_float3(k2));
	float4 k4 = tex3D(dataTex, loc3dTex+(dt/spacing)*make_float3(k3));
	return dt/6.0*(k1 + 2.0*(k2 + k3) + k4);
}


/*	Integrates (RK4) the vectorfield dataTex with timestep dt over
	steps number of timesteps, and saves the resulting curve in lineoutput. The w-coordinate
	of the output is unused.
	The parallelization assignes to each thread an initial value in the rectangle
	with corner startloc and sides xvec and yvec (modulo off-by-one). The output
	array lineoutput is linear, and consists of blocks of size 'steps' each containing 1 line
*/
__global__ void RK4line(float4* lineoutput, double dt, unsigned int steps, float4 startloc, float4 xvec, float4 yvec, dim3 gridsize) {
	dim3 index2D(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y*blockDim.y);
	float4 dx = (1.0/(gridsize.x*blockDim.x))*xvec;
	float4 dy = (1.0/(gridsize.y*blockDim.y))*yvec;
	float4 loc = startloc + index2D.x*dx + index2D.y*dy;
	int index = index2D.y*(gridsize.x*blockDim.x) + index2D.x;
	lineoutput[index*steps] = loc;
	for (unsigned int i=1; i < steps; i++) {
		loc = loc + RK4step(loc, dt);
		lineoutput[index*steps + i] = loc;
	}
	return;
}

/*	A testfunction that gives the value of the vectorfield dataTex along a line
	going in the positive x-direction for steps number of steps of size spacing (global)
	starting at loc. Saves the values in lineoutput
*/
__global__ void readline(float4* lineoutput, unsigned int steps, float4 loc) {
	float3 loc3d = make_float3(loc);
	for (unsigned int i=0; i < steps; i++) {
		lineoutput[i] = tex3D(dataTex, Smiet2Tex(loc));
		loc.x+=spacing;
	}
	return;
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

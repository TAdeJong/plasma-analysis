#include "coordfunctions.h"
#include "constants.h"
#include "integration.h"

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


/*	Integrates (RK4) the vectorfield dataTex with initial value loc and timestep dt over
	steps number of timesteps, and saves the resulting curve in lineoutput. The w-coordinate
	of the output is unused.
*/
__global__ void RK4line(float4* lineoutput, double dt, unsigned int steps, float4 loc) {
	lineoutput[0] = loc;
	for (unsigned int i=1; i < steps; i++) {
		loc = loc + RK4step(loc, dt);
		lineoutput[i] = loc;
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



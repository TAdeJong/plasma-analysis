#include "coordfunctions.h"
#include "constants.h"
#include "integration.h"

//Do 1 RK4 step. Return een waarde in Smietcoords, input in Smietcoords
__device__ float4 RK4step(float4 loc, double dt ) {
	float3 loc3dTex = Smiet2Tex(loc);
	float4 k1 = tex3D(dataTex, loc3dTex);
	float4 k2 = tex3D(dataTex, loc3dTex+(dt*0.5/spacing)*make_float3(k1));
	float4 k3 = tex3D(dataTex, loc3dTex+(dt*0.5/spacing)*make_float3(k2));
	float4 k4 = tex3D(dataTex, loc3dTex+(dt/spacing)*make_float3(k3));
	return dt/6.0*(k1 + 2.0*(k2 + k3) + k4);
}

__global__ void RK4line(float4* lineoutput, double dt, unsigned int steps, float4 loc) {
	lineoutput[0] = loc;
	for (unsigned int i=1; i < steps; i++) {
		loc = loc + RK4step(loc,dt);
		lineoutput[i] = loc;
	}
	return;
}

__global__ void readline(float4* lineoutput, unsigned int steps, float4 loc) {
	float3 loc3d = make_float3(loc);
	for (unsigned int i=0; i < steps; i++) {
		lineoutput[i] = tex3D(dataTex, Smiet2Tex(loc));
		loc.x+=spacing;
	}
	return;
}



#ifndef _INTEGRATION_H_
#define _INTEGRATION_H_

//Global texture
texture <float4, cudaTextureType3D, cudaReadModeElementType> dataTex;

__device__ float4 RK4step(float4 loc, double dt );
__global__ void RK4line(float4* lineoutput, double dt, unsigned int steps, float4 loc);
__global__ void readline(float4* lineoutput, unsigned int steps, float4 loc);

#endif

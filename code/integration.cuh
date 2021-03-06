/* Contains all functions relating to line integration (RK4 and a testfunction) */

#ifndef _INTEGRATION_H_
#define _INTEGRATION_H_

__global__ void RK4init(float* d_lengths, float4 startloc, float4 xvec, float4 yvec);
__device__ float4 RK4step(float4 loc, double dt);

__global__ void RK4line(float4* lineoutput, double dt, const unsigned int steps, float4 startloc, float4 xvec, float4 yvec);

__global__ void readline(float4* lineoutput, unsigned int steps, float4 loc);

__device__ float Lengthstep( float4 loc, double dt);

__global__ void lineLength(float4* g_linedata, double dt, float* g_lengthoutput);

__global__ void rxy(float4* g_linedata, float* radius, const float norm, float4* offset, float4* normal);

#endif

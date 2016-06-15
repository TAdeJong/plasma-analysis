/* Contains all functions relating to line integration (RK4 and a testfunction) */

#ifndef _INTEGRATION_H_
#define _INTEGRATION_H_

__device__ float4 RK4step(float4 loc, double dt);

__global__ void RK4line(float4* lineoutput, double dt, unsigned int steps, float4 startloc, float4 xvec, float4 yvec, dim3 gridsize);

__global__ void readline(float4* lineoutput, unsigned int steps, float4 loc);


#endif

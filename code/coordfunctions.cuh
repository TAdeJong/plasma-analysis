#ifndef _COORDFUNC_H_
#define _COORDFUNC_H_
//Definitions of vectortype operators
__device__ float4 tex3D(texture<float4, 3, cudaReadModeElementType> tex, float3 a);

__device__ float3 Smiet2Tex(float4 locSmiet);

__device__ float4 ShiftCoord(float4 locSmiet, float4 offset);

__device__ float4 RotateCoord(float4 locSmiet, float4 znew);

__device__ float4 Cart2Sphere(float4 locSmiet);

__device__ float4 Cart2Tor(float4 locSmiet, float r);

#endif

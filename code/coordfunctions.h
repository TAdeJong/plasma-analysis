#ifndef _COORDFUNC_H_
#define _COORDFUNC_H_
#include "constants.h"
//Definitions of vectortype operators
__device__ float3 operator+(const float3 &a, const float3 &b);

__device__ float3 operator*(const float &a, const float3 &b);

__device__ float4 operator+(const float4 &a, const float4 &b);

__device__ float4 operator*(const float &a, const float4 &b);

__device__ float3 make_float3(float4 a);

__device__ float4 tex3D(texture<float4, 3, cudaReadModeElementType> tex, float3 a);

__device__ float3 Smiet2Tex(float4 locSmiet);

#endif

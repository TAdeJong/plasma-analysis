/* 	Contains the functions relating to transforming the Cartesian coordinate system into
	donut coordinates, including finding the middle of a field line
*/

#ifndef _CONVERSIONS_H_
#define _CONVERSIONS_H_

__device__ float4 origin(float4* lineoutput, int steps, dim3 gridsize, int numberoflines, float4* communication);

__device__ float4 normal(float4* lineoutput, int steps, dim3 gridsize, int numberoflines, float4* communication, float4 origin);

__device__ void reducesum( float4* g_linedata, float4* g_sumdata);

__device__ void reducenormal( float4* g_linedata, float4* g_normaldata);
#endif

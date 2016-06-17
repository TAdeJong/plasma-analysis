/* 	Contains the functions relating to transforming the Cartesian coordinate system into
	donut coordinates, including finding the middle of a field line
*/

#ifndef _CONVERSIONS_H_
#define _CONVERSIONS_H_

__device__ float4 origin(float4* lineoutput, int steps, dim3 gridsize, int numberoflines, float4* communication)

__device__ float4 normal(float4* lineoutput, int steps, dim3 gridsize, int numberoflines, float4* communication, float4 origin)

#endif

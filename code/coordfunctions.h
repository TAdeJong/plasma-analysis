#ifndef _COORDFUNC_H_
#define _COORDFUNC_H_

const int N = 256;

const float spacing = 0.0245436930189;
const float origin = -3.12932085991;


//Definitions of vectortype operators
__device__ float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 operator*(const float &a, const float3 &b) {
	return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float4 operator+(const float4 &a, const float4 &b) {
	return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__device__ float4 operator*(const float &a, const float4 &b) {
	return make_float4(a*b.x, a*b.y, a*b.z,a*b.w);
}

inline __device__ float3 make_float3(float4 a) {
	return make_float3(a.x,a.y,a.z);
}

inline __device__ float4 tex3D(texture<float4, 3, cudaReadModeElementType> tex, float3 a) {
	return tex3D(tex, a.x,a.y,a.z);
}

inline __device__ float3 Smiet2Tex(float4 locSmiet) {
	return make_float3((locSmiet.x-origin)/spacing,(locSmiet.y-origin)/spacing,(locSmiet.z-origin)/spacing);
}

#endif

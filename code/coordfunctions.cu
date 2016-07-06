#include "constants.h"
#include "coordfunctions.h"
#include "constants.h"
#include "helper_math.h"

__device__ float4 tex3D(texture<float4, 3, cudaReadModeElementType> tex, float3 a) {
	return tex3D(tex, a.x,a.y,a.z);
}

/*	Converts a float4 of Smietcoordinates (values between origin (which is negative) and -origin)
	to a float3 of Texture-coordinates (values between 0.5 and N+0.5).
*/
__device__ float3 Smiet2Tex(float4 locSmiet) {
	return make_float3((locSmiet.x-origin)/spacing+0.5,(locSmiet.y-origin)/spacing+0.5,(locSmiet.z-origin)/spacing+0.5);
}

__device__ float4 ShiftCoord(float4 locSmiet, float4 offset) {
	return locSmiet - offset;
}

//Checked
__device__ float4 RotateCoord(float4 locSmiet, float4 znew, float4 ynew) {
	znew = normalize(znew);
	ynew = normalize(ynew);
	float4 xnew = make_float4(cross(make_float3(ynew), make_float3(znew)), 0); //rechtshandig
	return make_float4(dot(xnew,locSmiet),dot(ynew,locSmiet),dot(znew,locSmiet), 0);
}

//Checked
__device__ float4 Cart2Sphere(float4 locSmiet) {
	return make_float4(length(locSmiet),acos(locSmiet.z/length(locSmiet)),atan(locSmiet.y/locSmiet.x),0);
}

//Checked
__device__ float4 Cart2Tor(float4 locSmiet, float R) {
	float alpha = atan(locSmiet.y/locSmiet.x);
	float rho = length(make_float2(locSmiet.x,locSmiet.y));//projection onto xy-plane
	float beta = atan(locSmiet.z/(rho-R)); 
	float r = length(make_float2(rho-R,locSmiet.z));//in alpha=constant plane
	return make_float4(r,alpha,beta,0);
}


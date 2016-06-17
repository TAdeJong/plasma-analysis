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


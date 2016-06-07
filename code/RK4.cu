#include <stdio.h>
texture <float4, cudaTextureType3D, cudaReadModeElementType> dataTex;

int N = 256;
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

//Do 1 RK4 step.
__device__ float4 RK4step(float4 loc, double dt ) {
	float3 loc3d = make_float3(loc);
	float4 k1 = tex3D(dataTex, loc3d); // of moet ik hier loc.x,loc.y,loc.z meegeven?
	float4 k2 = tex3D(dataTex, loc3d+dt*0.5*make_float3(k1));
	float4 k3 = tex3D(dataTex, loc3d+dt*0.5*make_float3(k2));
	float4 k4 = tex3D(dataTex, loc3d+dt*make_float3(k3));
	return dt/6.0*(k1 + 2.0*(k2 + k3) + k4);
}

__global__ void RK4line(float4* lineoutput, double dt, unsigned int steps, float4 loc) {
	lineoutput[0] = loc;
	for (unsigned int i=1; i < steps; i++) {
		lineoutput[i] = lineoutput[i-1]+RK4step(lineoutput[i-1],dt);
	}
	return;
}


int main(void) {
	cudaArray* dataArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
	cudaMallocArray(&dataArray, &channelDesc,N,N,N);
	cudaMemcpyToArray(dataArray,0,0,hostpointer,N*N*N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(dataTex, dataArray, channelDesc);
	float4 *lines;
	int steps = 20;
	int cores = 1;
	int blocks = 1;
	float dt = 1/8.0;
	float4 startloc = {0};
	cudaMalloc(&lines,blocks*cores*steps*sizeof(float4));
	RK4line<<<cores,blocks>>>(lines, dt, steps, startloc);
	printf("Hello World!\n");
	return 0;
}

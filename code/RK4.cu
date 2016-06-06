#include <stdio.h>
texture <float4, cudaTextureType3D> dataTex;

int N = 256;
//Definitions of vectortype operators
__device__ float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 operator*(const float &a, const float3 &b) {
	return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float4 operator+(const float4 &a, const float4 &b) {
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__device__ float4 operator*(const float &a, const float4 &b) {
	return make_float3(a*b.x, a*b.y, a*b.z,a*b.w);
}

//Do 1 RK4 step.
__device__ float4 RK4step(float4 loc, double dt ) {
	float4 k1 = tex3D(dataTex, (float3)loc); // of moet ik hier loc.x,loc.y,loc.z meegeven?
	float4 k2 = tex3D(dataTex, (float3)loc+dt*0.5*k1);
	float4 k3 = tex3D(dataTex, (float3)loc+dt*0.5*k2);
	float4 k4 = tex3D(dataTex, (float3)loc+dt*k3);
	return (float4)dt/6.0*(k1 + 2.0*(k2 + k3) + k4);
}

__global__ void RK4line(float4& lineoutput, double dt, unsigned int steps, float4 loc) {
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
	cudaBindTextureToArray(dataTex, dataArray, channeldesc);
	float4 *lines;
	int steps = 20;
	int cores = 1;
	int blocks = 1;
	float dt = 1/8.0;
	float4 startloc = {{0}};
	cudaMalloc(&lines,blocks*cores*steps*sizeof(float4));
	RK4line<<<cores,blocks>>>(lines, dt, steps, startloc);
	printf("Hello World!\n");
	return 0;
}

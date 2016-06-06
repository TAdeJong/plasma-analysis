#include <stdio.h>
texture <float4, cudaTextureType3D> dataTex;

int N = 256;

__device__ float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 operator*(const float &a, const float3 &b) {
	return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ float4 RK4step(double3* output, double3 loc, double dt ) {
	float4 k1 = tex3D(dataTex, (float3)loc); // of moet ik hier loc.x,loc.y,loc.z meegeven?
	float4 k2 = tex3D(dataTex, (float3)loc+dt*0.5*k1);
	float4 k3 = tex3D(dataTex, (float3)loc+dt*0.5*k2);
	float4 k4 = tex3D(dataTex, (float3)loc+dt*k3);
	return dt/6.0*(k1 + 2.0*((double4)k2 + (double4)k3) + k4);
}


int main(void) {
	cudaArray* dataArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
	cudaMallocArray(&dataArray, &channelDesc,N,N,N);
	cudaMemcpyToArray(dataArray,0,0,hostpointer,cudaMemcpyHostToDevice);
	cudaBindTextureToArray(dataTex, dataArray, channeldesc);
	nullkernel<<<1,1>>>();
	printf("Hello World!\n");
	return 0;
}

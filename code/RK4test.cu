#include <stdio.h>
#include <iostream>
#include <helper_cuda.h>

texture <float4, cudaTextureType3D, cudaReadModeElementType> dataTex;

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

inline __device__ __host__ float3 Smiet2Tex(float4 locSmiet) {
	return make_float3((locSmiet.x-origin)/spacing+0.5,(locSmiet.y-origin)/spacing+0.5,(locSmiet.z-origin)/spacing+0.5);
}

//Do 1 RK4 step. Return een waarde in Smietcoords, input in Smietcoords
__device__ float4 RK4step(float4 loc, double dt ) {
	float3 loc3dTex = Smiet2Tex(loc);
	float4 k1 = tex3D(dataTex, loc3dTex);
	float4 k2 = tex3D(dataTex, loc3dTex+(dt*0.5/spacing)*make_float3(k1));
	float4 k3 = tex3D(dataTex, loc3dTex+(dt*0.5/spacing)*make_float3(k2));
	float4 k4 = tex3D(dataTex, loc3dTex+(dt/spacing)*make_float3(k3));
	return dt/6.0*(k1 + 2.0*(k2 + k3) + k4);
}

__global__ void RK4line(float4* lineoutput, double dt, unsigned int steps, float4 loc) {
	lineoutput[0] = tex3D(dataTex, Smiet2Tex(loc));
	for (unsigned int i=1; i < steps; i++) {
		loc = loc + RK4step(loc,dt);
		lineoutput[i] = loc;
	}
	return;
}

__global__ void readline(float4* lineoutput, unsigned int steps, float4 loc) {
	float3 loc3d = make_float3(loc);
	for (unsigned int i=0; i < steps; i++) {
		lineoutput[i] = tex3D(dataTex, Smiet2Tex(loc));
		loc.x+=spacing;
	}
	return;
}


void datagen (float4*** data) {
	//data[z][y][x]
	for (int i=0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				(data[i][j][k]).x = - (origin + spacing*j);
				(data[i][j][k]).y = (origin + spacing*k);
				(data[i][j][k]).z = 0;
				(data[i][j][k]).w = 0;
			}
		}
	}
}

int main(void) {
	cudaArray* dataArray;
//	std::cout << 1 << std::endl;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
	cudaExtent extent = make_cudaExtent(N , N, N);
	checkCudaErrors(cudaMalloc3DArray(&dataArray, &channelDesc,extent));

	dataTex.filterMode = cudaFilterModeLinear;
	//Generate data
	float4*** hostvfield; 
//	hostvfield = (float4*) malloc(N*N*N*sizeof(float4));
	hostvfield = (float4***) malloc(N*sizeof(float4**));
	hostvfield[0] = (float4**) malloc(N*N*sizeof(float4*));
	hostvfield[0][0] = (float4*) malloc(N*N*N*sizeof(float4));
//	std::cout << 2 << std::endl;
	for (int i=1; i < N; i++) {
		hostvfield[i] = (hostvfield[0] + i*N);
	}
	for (int i=0; i < N; i++) {
		for (int j=0; j < N; j++) {
			hostvfield[i][j] = (hostvfield[0][0] + (i*N*N + j*N));
		}
	}
//	std::cout << 2 << std::endl;
	datagen(hostvfield);
//	std::cout << extent.width << extent.height << extent.depth << std::endl;
//copy to device
	cudaMemcpy3DParms copyParms = {0};
	copyParms.srcPtr = make_cudaPitchedPtr((void *)hostvfield[0][0], extent.width* sizeof(float4), extent.height, extent.depth);
//	std::cout << 4 << std::endl;
	copyParms.dstArray = dataArray;
	copyParms.extent = extent;
	copyParms.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParms));
//	std::cout << 5 << std::endl;
	checkCudaErrors(cudaBindTextureToArray(dataTex, dataArray, channelDesc));
	float4 *d_lines, *h_lines;
	double time = 3.141592653*2.0;
	int steps = 1000;
	int cores = 1;
	int blocks = 1;
	float dt = time/N;
	float4 startloc = {1,0,0,0};
	float3 locSmiet = Smiet2Tex(startloc);
	std::cout << "Starting point in texture coordinates: x=" << locSmiet.x << ", y=" << locSmiet.y << ", z=" << locSmiet.z << std::endl;
//	std::cout << hostvfield[127][127][127].x << std::endl;
	checkCudaErrors(cudaMalloc(&d_lines,blocks*cores*steps*sizeof(float4)));

	h_lines = (float4*) malloc(blocks*cores*steps*sizeof(float4));
	RK4line<<<cores,blocks>>>(d_lines, dt, steps, startloc);
	checkCudaErrors(cudaMemcpy(h_lines, d_lines, blocks*cores*steps*sizeof(float4), cudaMemcpyDeviceToHost));
	for(unsigned int i=0; i<3; i++) {
		std::cout << "x= " << h_lines[i].x << "; y= "<< h_lines[i].y << " "<< h_lines[i].x*h_lines[i].x+h_lines[i].y*h_lines[i].y << std::endl;
	}
	free(hostvfield[0][0]);
	free(hostvfield[0]);
	free(hostvfield);
	
	return 0;
}


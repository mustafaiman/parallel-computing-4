#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LONG long long int
#define SIZE_PER_THREAD 256

int *generate_random_array(int n) {
	int *arr = (int *)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		arr[i] = rand() % 100;
	}
	return arr;
}

LONG cpu_dot_product(int *arr1, int *arr2, int n) {
	LONG dot_product = 0;
	for (int i = 0; i < n; i++) {
		dot_product += arr1[i] * arr2[i];
	}
	return dot_product;
}

__global__ void kernel_dot_product(int *arr1, int *arr2, int n) {
	extern __shared__ int partialSum[];
	int t = (blockIdx.x * blockDim.x + threadIdx.x ) * SIZE_PER_THREAD;
	partialSum[threadIdx.x] = 0;
	for (int i = 0; i < SIZE_PER_THREAD; i++)
	{
		if(t+i < n) {
			partialSum[threadIdx.x] += arr1[t+i] * arr2[t+i];
		}
	}
	for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
		__syncthreads();
		if (threadIdx.x < stride && t + stride < n) {
			partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
		}
	}
	//arr1[blockIdx.x] = partialSum[0];
}

int gpu_dot_product(int *arr1, int *arr2, int n, int blocksize) {
	int *d_arr1;
	int *d_arr2;
	int *d_n;

	printf("Number of elements: %d\n", n);
	printf("Number of threads per block: %d\n", blocksize);
	int num_of_blocks = (n + blocksize * SIZE_PER_THREAD - 1) / (blocksize * SIZE_PER_THREAD);
	printf("Number of blocks will be created: %d\n", num_of_blocks);

	cudaMalloc((void **)&d_arr1, n * sizeof(int));
	cudaMalloc((void **)&d_arr2, n * sizeof(int));

	cudaMemcpy(d_arr1, arr1, n* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arr2, arr2, n* sizeof(int), cudaMemcpyHostToDevice);

	kernel_dot_product <<< num_of_blocks ,blocksize, blocksize * sizeof(int) >>>(d_arr1, d_arr2, n);
	cudaThreadSynchronize();

	cudaMemcpy(arr1, d_arr1, 5* sizeof(int), cudaMemcpyDeviceToHost);

	return 0;
}

int main(int argc, char *argv[]) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	srand(time(NULL));

	int n = atoi(argv[1]);
	int blocksize = atoi(argv[2]);

	double milliseconds = clock();
	int *arr1 = generate_random_array(n);
	int *arr2 = generate_random_array(n);
	milliseconds = (clock() - milliseconds)/(double)CLOCKS_PER_SEC * 1000;

	printf("Time array generation: %lf\n", milliseconds);

	LONG cpu_result;
	milliseconds = clock();
	cpu_result = cpu_dot_product(arr1, arr2, n);
	milliseconds = (clock() - milliseconds)/(double)CLOCKS_PER_SEC * 1000;
	printf("Cpu result: %lld\n", cpu_result);	

	printf("Time cpu: %lf\n", milliseconds);

	int gpu_result;
	cudaEventRecord(start);
	gpu_result = gpu_dot_product(arr1, arr2, n, blocksize);
	cudaEventRecord(stop);
	printf("Gpu result: %d\n", gpu_result);

	float cudaTime = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cudaTime, start, stop);

	printf("Time GPU: %f ms\n", cudaTime);
	
	free(arr1);
	free(arr2);
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LONG long long int
#define SIZE_PER_THREAD 256

double interval_host_to_device;
double interval_device_to_host;

float interval_kernel;

int *generate_random_array(int n) {
	int *arr = (int *)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		arr[i] = rand();
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

__global__ void kernel_dot_product(int *arr1, int *arr2, int n, LONG *output) {
	extern __shared__ LONG partialSum[];
	int t = (blockIdx.x * blockDim.x + threadIdx.x ) * SIZE_PER_THREAD;
	LONG localSum = 0;
	for (int i = 0; i < SIZE_PER_THREAD; i++)
	{
		if(t+i < n) {
			localSum += arr1[t+i] * arr2[t+i];
		}
	}
	partialSum[threadIdx.x] = localSum;
	for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
		__syncthreads();
		if (threadIdx.x < stride && t + stride < n) {
			partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
		}
	}
	if (threadIdx.x == 0) {
		output[blockIdx.x] = partialSum[0];
	}
}

LONG gpu_dot_product(int *arr1, int *arr2, int n, int blocksize, int num_of_blocks) {
	int *d_arr1;
	int *d_arr2;
	LONG *d_n;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaMalloc((void **)&d_arr1, n * sizeof(int));
	cudaMalloc((void **)&d_arr2, n * sizeof(int));
	cudaMalloc((void **)&d_n, num_of_blocks * sizeof(LONG));

	interval_host_to_device = clock();
	cudaMemcpy(d_arr1, arr1, n* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arr2, arr2, n* sizeof(int), cudaMemcpyHostToDevice);
	interval_host_to_device = (clock() - interval_host_to_device) / (double)CLOCKS_PER_SEC * 1000;

	cudaEventRecord(start);


	kernel_dot_product << < num_of_blocks, blocksize, blocksize * sizeof(LONG) >> >(d_arr1, d_arr2, n, d_n);


	cudaEventRecord(stop);
	cudaThreadSynchronize();

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&interval_kernel, start, stop);


	LONG *block_output = (LONG *)malloc(num_of_blocks * sizeof(LONG));
	interval_device_to_host = clock();
	cudaMemcpy(block_output, d_n, num_of_blocks * sizeof(LONG), cudaMemcpyDeviceToHost);
	interval_device_to_host = clock() - interval_device_to_host;

	LONG sum = 0;
	for (int i = 0; i < num_of_blocks; i++) {
		sum += block_output[i];
	}

	return sum;
}

int main(int argc, char *argv[]) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	srand(time(NULL));

	int n = atoi(argv[1]);
	int blocksize = atoi(argv[2]);

	printf("Info\n------------\nNumber of elements: %d\n", n);
	printf("Number of threads per block: %d\n", blocksize);	
	int num_of_blocks = (n + blocksize * SIZE_PER_THREAD - 1) / (blocksize * SIZE_PER_THREAD);
	printf("Number of blocks will be created: %d\n", num_of_blocks);

	printf("Time\n-------\n");

	double interval_array_generation = clock();
	int *arr1 = generate_random_array(n);
	int *arr2 = generate_random_array(n);
	interval_array_generation = (clock() - interval_array_generation)/(double)CLOCKS_PER_SEC * 1000;


	LONG cpu_result;
	double interval_cpu = clock();
	cpu_result = cpu_dot_product(arr1, arr2, n);
	interval_cpu = (clock() - interval_cpu) / (double)CLOCKS_PER_SEC * 1000;


	LONG gpu_result;
	cudaEventRecord(start);
	gpu_result = gpu_dot_product(arr1, arr2, n, blocksize, num_of_blocks);
	cudaEventRecord(stop);

	float cudaTime = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cudaTime, start, stop);

	printf("Time array generation: %lf\n", interval_array_generation);
	printf("Time for the Cpu function: %lf\n", interval_cpu);
	printf("Time for the Host to Device transfer: %lf\n", interval_host_to_device);
	printf("Time for the kernel execution: %lf\n", interval_kernel);
	printf("Time for the Device to Host transfer: %lf\n", interval_device_to_host);
	printf("Total execution time for GPU GPU: %f ms\n", cudaTime);



	printf("Results\n----------\nCpu result: %lld\n", cpu_result);

	printf("Gpu result: %lld\n", gpu_result);

	free(arr1);
	free(arr2);
}

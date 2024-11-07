/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
/**
 * We use the curand library on the **device**: we want to generate random numbers on the device
 * on the fly. Using the host library "curand.h", we are obliged to store the random numbers on the
 * global memory of the device, which is not efficient for our purpose( we want to generate the random
 * number and use it directly in the kernel).
 */
#include <curand_kernel.h>



// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x) {
	const double p = 0.2316419;
	const double b1 = 0.319381530;
	const double b2 = -0.356563782;
	const double b3 = 1.781477937;
	const double b4 = -1.821255978;
	const double b5 = 1.330274429;
	const double one_over_twopi = 0.39894228;
	double t;

	if (x >= 0.0) {
		t = 1.0 / (1.0 + p * x);
		return (1.0 - one_over_twopi * exp(-x * x / 2.0) * t * (t * (t *
			(t * (t * b5 + b4) + b3) + b2) + b1));
	}
	else {/* x < 0 */
		t = 1.0 / (1.0 - p * x);
		return (one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t *
			(t * b5 + b4) + b3) + b2) + b1));
	}
}

// Set the state for each thread
__global__ void init_curand_state_k(curandState_t* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(0, idx, 0, &state[idx]);
}


// Monte Carlo simulation kernel
// dt = sqrt(T/N)
// dt*dt= T/N
__global__ void MC_k1(float S_0, float r, float sigma, float dt, float K,
						int N, curandState_t* state, float* payGPU)
{

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx]; // in this way we avoid two different series  to be the same
	float2 G;
	float S = S_0;

	for(int i = 0; i<N; i++)
	{
		G = curand_normal2(&localState);
		S *= (1 + r * dt * dt + sigma * dt * G.x);
	}

	payGPU[idx] = expf(-r * dt *dt * N) * fmaxf(0.0f, S - K);

	// if am doing only one montecarlo simulation
	// I haeve to begin again the sequence
	// state[idx] = localState;
}

// we now need n, 
__global__ void MC_k2(float S_0, float r, float sigma, float dt, float K,
						int N, curandState_t* state, float* sum, int n)
{

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx]; // in this way we avoid two different series  to be the same
	float2 G;
	float S = S_0;
	extern __shared__ float A[]; // dynamically allocated in the kernel call
	float* R1s, * R2s; 
	R1s = A;
	R2s = R1s + blockDim.x;

	for(int i = 0; i<N; i++)
	{
		G = curand_normal2(&localState);
		S *= (1 + r * dt * dt + sigma * dt * G.x);
	}

	R1s[threadIdx.x] = expf(-r * dt *dt * N) * fmaxf(0.0f, S - K)/n;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x]/n;

	__syncthreads(); // wait for all threads to finish the computation

	int i = blockDim.x/2;
	while(i != 0)
	{
		if(threadIdx.x < i)
		{
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads(); // wait for all threads to finish the computation
		i /= 2;
	}

	if(threadIdx.x == 0)
	{
		atomicAdd(sum, R1s[0]);
		atomicAdd(sum +1, R2s[0]);
	}

	// if am doing only one montecarlo simulation
	// I haeve to begin again the sequence
	// state[idx] = localState;
}

int main(void) {

  // 512 * 512 takes 0.700448 ms

	int NTPB = 1024;
	int NB = 512;
	int n = NB * NTPB;
	float T = 1.0f;
	float S_0 = 50.0f;
	float K = S_0;
	float sigma = 0.2f;
	float r = 0.1f;
	int N = 100;
	float dt = sqrtf(T/N);
	float *sum;

	// the main idea is to have each threads to store is own state
	// in order to avoid the use of communication and each thread can 
	// execute the kernel independently
	printf("%zd\n", sizeof(curandStateXORWOW_t)); // default
	printf("%zd\n", sizeof(curandStateMRG32k3a_t)); // quite efficient and we can use with a 64 machine 
	printf("%zd\n", sizeof(curandStatePhilox4_32_10_t)); // used by pytorch (a little bit heavier)
	// very big type, IT SHOULD NOT BE USED otherwise we finish fast the local memory
	printf("%zd\n", sizeof(curandStateMtgp32_t)); 

	cudaMallocManaged(&sum, 2 * sizeof(float));
	cudaMemset(sum, 0, 2 * sizeof(float));

	/*
	 *  Define the arrays of the actualized payoffs and perform the right copy.
	 */
	// float *payGPU, *payCPU;
	// malloc the array payoffs
	// payCPU = (float*)malloc(n * sizeof(float));
	// cudaMalloc the array payoffs 
	//cudaMalloc(&payGPU, n * sizeof(float));

	curandState_t* state;
	// cudaMalloc the array state
	cudaMalloc(&state, n * sizeof(curandState_t)); // is the total number of state

	init_curand_state_k<<<NB, NTPB>>>(state);

	float Tim;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	//MC_k1<<<NB, NTPB>>>(S_0, r, sigma, dt, K, N, state, payGPU);

	// in this new version we include also the 
	// reduction on the GPU and the data transfer
	MC_k2 < << NB, NTPB, 2*NTPB*sizeof(float)>> > ( S_0, r, sigma, dt, K, N, state, sum, n);

	cudaDeviceSynchronize(); // since now we do not perform the cuda memcopy and we 
	
	// the last block 

	// GPU has its own scheduler so if we do the
                              // the memcpy, the GPU will finish the kernel before

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

	// Copy the payoffs from the device to the host 
	//cudaMemcpy(payCPU, payGPU, n * sizeof(float), cudaMemcpyDeviceToHost);

	// Reduction performed on the host
	/*for (int i = 0; i < n; i++) {
		sum += payCPU[i]/n; // we compute here the division by n in order to avoid overflow
		sum2 += payCPU[i] * payCPU[i]/n;
	}*/

	// NOW THE SUM IS CARRIED ON THE CPU
	// sum[0] sum of the prices
	// sum[1] square of the sum
	printf("The estimated price is equal to %f\n", sum[0]);
	printf("error associated to a confidence interval of 95%% = %f\n",
		1.96 * sqrt((double)(1.0f / (n - 1)) * (n*sum[1] - (sum[0] * sum[0])))/sqrt((double)n));
	printf("The true price %f\n", S_0 * NP((r + 0.5 * sigma * sigma)/sigma) -
									K * expf(-r) * NP((r - 0.5 * sigma * sigma) / sigma));
	printf("Execution time %f ms\n", Tim);

	// Free the memory
	//cudaFree(payGPU);

	return 0;
}
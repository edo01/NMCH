#include <curand_kernel.h>
#include <cmath>
#include <iostream>

const float K = 1.0f;     
const float S0 = 1.0f;    // the spot values
const float v0 = 0.1f;  
const float r = 0.0f;     // the risk-free interest rate
const float kappa = 0.5f; // the mean reversion rate of the volatility
const float theta = 0.1f; // the long-term volatility
const float sigma = 0.3f; // the volatility of volatility
const float rho = -0.7f;  
const int T = 1;          
const int steps = 1000;   
const float dt = 1.0f / steps; 
const int simulations = 100000; 

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
__global__ void init_curand_state(curandState_t* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(0, idx, 0, &state[idx]);
}

__global__ void hestonMonteCarlo(float *d_results, int steps, float dt, float kappa, float theta, float sigma, float rho, curandState_t* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //Initialize the random number generator
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];

    float St = S0;
    float vt = v0;

    //Simulation time step
    for (int i = 0; i < steps; ++i) {
        float G1 = curand_normal(&localState);
        float G2 = curand_normal(&localState);

        // Calculate the delta of asset price and volatility
        float dSt = r * St * dt + sqrtf(vt) * St * sqrtf(dt) * (rho * G1 + sqrtf(1 - rho * rho) * G2);
        float dvt = kappa * (theta - vt) * dt + sigma * sqrtf(vt) * sqrtf(dt) * G1;

        St += dSt;
        vt = fabs(vt + dvt); // the function g is either taken to be equal to (·)+ or to | · |
    }
    // E[f(ST )] = E[(S1 − 1)+].
    d_results[tid] = fmaxf(St - K, 0.0f);
}


//Gamma_distribution
__device__ float gamma_distribution(float alpha, curandState* state) {
    if (alpha >= 1.0f) {
        float d = alpha - 1.0f / 3.0f;
        float c = 1.0f / sqrtf(9.0f * d);
        float x, v, u;

        while (true) {
            x = curand_normal(state);
            v = 1.0f + c * x;
            v = v * v * v;
            if (v <= 0.0f) continue;
            u = curand_uniform(state);
            if (u < 1.0f - 0.0331f * (x * x) * (x * x)) return d * v;
            if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v;
        }
    } else {
        //A random variable X ~ G(alpha + 1, beta), then: Y = X * U^{1/alpha}, then Y ~G(alpha, beta).
        float u = curand_uniform(state);
        return gamma_distribution(alpha + 1.0f, state) * powf(u, 1.0f / alpha);
    }
}

__global__ void exact_simulation(float *d_results_exact, int steps, float dt, float kappa, float theta, float sigma, float rho, curandState_t* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[tid];

    float St = S0;
    float vt = v0;
    float vI = 0.0f;
    float v1 = 0.0f;

    for (int i = 0; i < steps; ++i) {
        float d = 2.0f * kappa * theta / (sigma * sigma);
        float lambda = (2 * kappa * expf(-kappa * dt) * vt) / (sigma * sigma * (1 - expf(-kappa * dt)));
        int N = curand_poisson(&localState, lambda);
        float gamma = gamma_distribution(d + N, &localState);
        
        float vt_next = (sigma * sigma * (1.0f - expf(-kappa * dt)) / (2.0f * kappa)) * gamma;

        vI += 0.5f * (vt + vt_next) * dt;

        vt = vt_next;

        if (i == 1) v1 = vt;  
        
    }

    float integral_W = (1.0f / sigma) * (v1 - v0 - kappa * theta + kappa * vI);
    float m = -0.5f * vI + rho * integral_W;
    float sigma2 = (1.0f - rho * rho) * vI;
    float exponent = m + sqrtf(sigma2) * curand_normal(&localState);
    St = expf(exponent);
    d_results_exact[tid] = fmaxf(St - K, 0.0f);
}



int main() {
    
    int NTPB = 256;
    int NB = (simulations + NTPB - 1) / NTPB;
    
    //Allocate memory on the device to store the results
    float *d_results;
    float *d_results_exact;
    cudaMalloc((void **)&d_results, simulations * sizeof(float));
    cudaMalloc((void **)&d_results_exact, simulations * sizeof(float));

    curandState_t* state;
	// cudaMalloc the array state
	cudaMalloc(&state, simulations * sizeof(curandState_t)); // is the total number of state

    init_curand_state<<<NB, NTPB>>>(state);

	float Tim;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

    hestonMonteCarlo<<<NB, NTPB>>>(d_results, steps, dt, kappa, theta, sigma, rho, state);

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

    float *h_results = (float *)malloc(simulations * sizeof(float));
    cudaMemcpy(h_results, d_results, simulations * sizeof(float), cudaMemcpyDeviceToHost);

    float option_price = 0.0f;
    for (int i = 0; i < simulations; ++i) {
        option_price += h_results[i];
    }
    option_price /= simulations;
    option_price *= expf(-r * T); 

    printf("The estimated price is equal to %f\n", option_price);

    float Tim1;
	cudaEvent_t start1, stop1;			// GPU timer instructions
	cudaEventCreate(&start1);			// GPU timer instructions
	cudaEventCreate(&stop1);				// GPU timer instructions
	cudaEventRecord(start1, 0);			// GPU timer instructions

    exact_simulation<<<NB, NTPB>>>(d_results_exact, steps, dt, kappa, theta, sigma, rho, state);
	
    cudaEventRecord(stop1, 0);			// GPU timer instructions
	cudaEventSynchronize(stop1);			// GPU timer instructions
	cudaEventElapsedTime(&Tim1,			// GPU timer instructions
		start1, stop1);					// GPU timer instructions
	cudaEventDestroy(start1);			// GPU timer instructions
	cudaEventDestroy(stop1);				// GPU timer instructions

    float *h_results_exact = (float *)malloc(simulations * sizeof(float));
    cudaMemcpy(h_results_exact, d_results_exact, simulations * sizeof(float), cudaMemcpyDeviceToHost);

    float option_price_exact = 0.0f;
    for (int i = 0; i < simulations; ++i) {
        option_price_exact += h_results_exact[i];
    }
    option_price_exact /= simulations;
    option_price_exact *= expf(-r * T); 

    printf("The estimated price is equal to %f\n", option_price_exact);


	printf("The true price %f\n", S0 * NP((r + 0.5 * sigma * sigma)/sigma) -
									K * expf(-r) * NP((r - 0.5 * sigma * sigma) / sigma));

	printf("Execution time %f ms for heston\n", Tim);
    printf("Execution time %f ms for exact\n", Tim1);
    free(h_results);
    cudaFree(d_results);
    free(h_results_exact);
    cudaFree(d_results_exact);

    return 0;
}






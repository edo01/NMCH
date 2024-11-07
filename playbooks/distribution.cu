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
    float d, U, v, x, c;

    d = alpha - 1.0 /3.0; 
    c = 1.0 / sqrtf(9*d);
    
    while(alpha>=1){
        x = curand_normal(state); //Normal variate;
        U = curand_uniform(state); // Uniform variate
        v = pow((1 + x * c),3); 
        while(v<=0){
            v = pow((1 + x * c),3); 
        }
        // log(U) < 0.5 * x^2 + d - 2
        if (logf(U) < 0.5 * x * x + d - 2) {
            return d*v;
        }
        if( U < 1.0 - 0.0331 * (x*x)*(x*x)){
            return (d*v);
        } 
    }
}


__device__ float gamma_distribution2(float alpha, curandState *state) {
    if (alpha >= 1.0f) {
        float d = alpha - 1.0f / 3.0f;
        float c = 1.0f / sqrt(9.0f * d);
        while (true) {
            float x = curand_normal(state);  
            float v = 1.0f + c * x;
            v = v * v * v;
            
            float u = curand_uniform(state);
            if (u < 1.0f - 0.0331f * x * x * x * x || log(u) < 0.5f * x * x + d - 2.0f) {
                return d * v;  
            }
        }
    } else {
        // alpha < 1 
        while (true) {
            float u = curand_uniform(state);
            float x = curand_exponential(state);
            if (u <= pow(x, alpha)) {
                return x;  
            }
        }
    }
}

__global__ void exact_simulation(float *d_results_exact, int steps, float dt, float kappa, float theta, float sigma, float rho, curandState_t* state) {

    float St = S0;  
    float vt = v0;  
    float vI = 0.0f;  
    float v1 = 0.0f;
    float m = 0.0f;   
    float sigma2 = 0.0f;
    float lambda;
    float d;
    float gamma;
    int N;
    float integral;

    // initialization
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState = state[idx];

    for (int i = 0; i < steps; ++i) {
        //step 1
        d = 2 * kappa * theta/(sigma * sigma);
        for (int i = 0; i < steps; i++) {
            // Calculate lambda for Poisson distribution
            lambda = (2.0f * kappa * exp(-kappa * dt) * vt) / (sigma * sigma) * (1.0f - exp(-kappa * dt));
            N = curand_poisson(&localState,lambda);// Simulate Poisson process
            gamma = gamma_distribution2(N + d, &localState);// Simulate Gamma distribution 
            
            vt = sigma * sigma * (1.0f - exp(-kappa * dt)) / (2.0f * kappa) * gamma;
            
            if(i == 1){
                v1 = vt;
            }

            //step 2
            vI += 0.5f * (vt + vI);
            
            //step 3
            integral = 1.0 / sigma * (v1 - v0 - kappa * theta + kappa * vI);
            m += -0.5f * vI + rho * integral;
            sigma2 = (1.0f - rho * rho) * vI;
            St = exp(m + sigma2 * curand_normal(&localState));
        }

        // Payoff : (S_T - K)+
        float payoff = fmax(St - 1.0f, 0.0f);  
        d_results_exact[idx] = payoff;
    }
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

    printf("The exact estimated price is equal to %f\n", option_price_exact);


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






#include <curand_kernel.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

const float K = 1.0f;     
const float S0 = 1.0f;    
const float v0 = 0.1f;    
const float r = 0.0f;     
const float rho = -0.7f;  
const int T = 1;          
const int steps = 1000;   
const float dt = 1.0f / steps; 
const int simulations = 100000; 

void testCUDA(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("Error in file %s at line %d\n", file, line);
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

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
        return (1.0 - one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
    }
    else {
        t = 1.0 / (1.0 - p * x);
        return (one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
    }
}

__global__ void init_curand_state(curandState_t* state) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(0, idx, 0, &state[idx]);
}

__global__ void hestonMonteCarlo(float *d_results, int steps, float dt, float kappa, float theta, float sigma, float rho, curandState_t* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[tid];

    float St = S0;
    float vt = v0;

    for (int i = 0; i < steps; ++i) {
        float G1 = curand_normal(&localState);
        float G2 = curand_normal(&localState);

        float dSt = r * St * dt + sqrtf(vt) * St * sqrtf(dt) * (rho * G1 + sqrtf(1 - rho * rho) * G2);
        float dvt = kappa * (theta - vt) * dt + sigma * sqrtf(vt) * sqrtf(dt) * G1;

        St += dSt;
        vt = fabs(vt + dvt);
    }
    d_results[tid] = fmaxf(St - K, 0.0f);
}

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

    /* By selecting representative samples within the parameter range instead of 
     * stepping through all combinations, we can reduce computational overhead while 
     * observing the impact of different parameter values ​​on execution time, 
     * thereby avoiding the long simulation time caused by too many test combinations.
    */
    std::vector<float> kappas = {0.1f, 0.5f, 1.0f, 2.0f, 5.0f};
    std::vector<float> thetas = {0.01f, 0.1f, 0.2f, 0.3f, 0.5f};
    std::vector<float> sigmas = {0.1f, 0.3f, 0.5f, 0.7f, 1.0f};

    std::vector<float> euler_times;
    std::vector<float> exact_times;

    for (float kappa : kappas) {
        for (float theta : thetas) {
            for (float sigma : sigmas) {
                if (20 * kappa * theta > sigma * sigma) {
                    float *d_results, *d_results_exact;
                    cudaMalloc((void **)&d_results, simulations * sizeof(float));
                    cudaMalloc((void **)&d_results_exact, simulations * sizeof(float));

                    curandState_t* state;
                    cudaMalloc(&state, simulations * sizeof(curandState_t));
                    init_curand_state<<<NB, NTPB>>>(state);

                    //Euler
                    float eulerTime;
                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);
                    cudaEventRecord(start, 0);
                    hestonMonteCarlo<<<NB, NTPB>>>(d_results, steps, dt, kappa, theta, sigma, rho, state);
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&eulerTime, start, stop);
                    euler_times.push_back(eulerTime);
                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);

                    //exact simulation
                    float exactTime;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);
                    cudaEventRecord(start, 0);
                    exact_simulation<<<NB, NTPB>>>(d_results_exact, steps, dt, kappa, theta, sigma, rho, state);
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&exactTime, start, stop);
                    exact_times.push_back(exactTime);
                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);

                    cudaFree(d_results);
                    cudaFree(d_results_exact);
                    cudaFree(state);
                }
            }
        }
    }

    std::ofstream file("times.txt");
    for (size_t i = 0; i < euler_times.size(); ++i) {
        file << euler_times[i] << "," << exact_times[i] << "\n";
    }
    file.close();

    return 0;
}

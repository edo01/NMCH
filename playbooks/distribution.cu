#include <curand_kernel.h>
#include <cmath>
#include <iostream>

//Gamma_distribution
__device__ float gamma_distribution(float alpha, curandState* state) {
    float d, U, v, x, c;

    d = alpha - 1.0 /3.0; //d = a - 1/3
    c = 1.0 / sqrtf(9*d);
    
    while(alpha>=1){
        x = curand_normal(state); //Normal variate;
        U = curand_uniform(state); // Uniform variate
        v = pow((1 + x * c),3); 
        
        // log(U) < 0.5 * x^2 + d - 2
        if (v > 0 && logf(U) < 0.5 * x * x + d - 2) {
            return d*v;
        }
        if( U < 1.0 - 0.0331 * (x*x)*(x*x)){
            return (d*v);
        } 
    }
}



__global__ void exact_simulation(int num_simulations, float* results, float r, float kappa, float theta, float sigma,
                                        float S0, float v0, float T, int n_steps) {

    float St = S0;  
    float vt = v0;  
    float vI = 0.0f;  
    float v1 = 0.0f;
    float dt = T / n_steps;  
    float m = 0.0f;   
    float sigma2 = 0.0f;
    float lambda;
    float d;
    float gamma;
    float integral;

    // initialization
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState = state[idx];

    for (int i = 0; i < num_simulations; ++i) {
        
        d = 2 * kappa * theta/(sigma * sigma);
         // Calculate lambda for Poisson distribution
        for (int i = 0; i < n_steps; i++) {
            lambda = (2.0f * kappa * exp(-kappa * dt) * vt) / (sigma * sigma) * (1.0f - exp(-kappa * dt));
            
            // Simulate Poisson process
            int N = curand_poisson(state);

            // Simulate Gamma distribution 
            gamma = gamma_distribution(N + d, &state);
            
            vt = sigma * sigma * (1.0f - exp(-kappa * dt)) / (2.0f * kappa) * gamma;
            if(i==1){
                v1 = vt;
            }
            vI += 0.5f * (vt + vI);
            
            integral = 1.0 / sigma * (v1 - v0 - kappa * theta + kappa * vI);
            m += -0.5f * vI + rho * integral;
            sigma2 = (1.0f - rho * rho) * vI;

            St = exp(m + sigma2 * curand_normal(&state));
        }

        // Payoff : (S_T - K)+
        float payoff = fmax(St - 1.0f, 0.0f);  
        results[idx] = payoff;
    }
}






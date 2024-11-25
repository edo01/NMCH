#include "NMCH/methods/NMCH_EM.hpp"

#define testCUDA(error) (nmch::utils::cuda::checkCUDA(error, __FILE__ , __LINE__))


namespace nmch::methods::kernels{
    
    template <typename rnd_state>
    __device__
    float gamma_distribution(float alpha, rnd_state* state) 
    {
        float d, c, x, v, u;
        
        // big divercence in this case

        if (alpha >= 1.0f) {
            d = alpha - 1.0f / 3.0f;
            c = 1.0f / sqrtf(9.0f * d);

            while (true) {
                x = curand_normal(state);
                v = 1.0f + c * x;
                v = v * v * v;
                if (v <= 0.0f) continue;
                u = curand_uniform(state);
                if (u < 1.0f - 0.0331f * (x * x) * (x * x)) return d * v;
                if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v;
            }
        } else { // case alpha < 1
            // we are using the fact gamma_alpha = gamma_{alpha + 1} * U^{1/alpha}
            // 
            //A random variable X ~ G(alpha + 1, beta), then: Y = X * U^{1/alpha}, then Y ~G(alpha, beta).
            u = curand_uniform(state);
            alpha += 1.0f; // alpha + 1

            d = alpha - 1.0f / 3.0f;
            c = 1.0f / sqrtf(9.0f * d);

            while (true) {
                x = curand_normal(state);
                v = 1.0f + c * x;
                v = v * v * v;
                if (v <= 0.0f) continue;
                u = curand_uniform(state);
                if (u < 1.0f - 0.0331f * (x * x) * (x * x)) return d * v * powf(u, 1.0f / alpha);
                if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v * powf(u, 1.0f / alpha);
            }
        }
    }

    template <typename rnd_state>
    __global__ 
    /* void EM_K1(float *d_results_exact, int steps, float dt, float kappa, 
            float theta, float sigma, float rho, rnd_state* state)  */
    void EM_k1(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, rnd_state* state, float* sum, int n)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        extern __shared__ float A[]; // dynamically allocated in the kernel call
        rnd_state localState = state[tid];

        int i;

        int N_p; // poisson 

        // initialization of the volatility and the price
        float St = S_0;
        float vt = v_0;

        float vI = 0.0f;

        float lambda, gamma, vt_next;

        const float d = 2.0f * k * theta / (sigma * sigma);

        float *R1s, * R2s; 
        R1s = A;
        R2s = R1s + blockDim.x;

        for (i = 0; i < N; ++i) {
            lambda = (2 * k * expf(-k * dt) * vt) / (sigma * sigma * (1 - expf(-k * dt)));
            N_p = curand_poisson(&localState, lambda);
            gamma = gamma_distribution(d + N_p, &localState);
            
            vt_next = (sigma * sigma * (1.0f - expf(-k * dt)) / (2.0f * k)) * gamma;

            vI += 0.5f * (vt + vt_next) * dt;

            vt = vt_next;
        }
        //vt = v1;

        float integral_W = (1.0f / sigma) * (vt - v_0 - k * theta + k * vI);
        float m = -0.5f * vI + rho * integral_W;
        float sigma2 = (1.0f - rho * rho) * vI;
        St = expf(m + sqrtf(sigma2) * curand_normal(&localState));

        // reduction

        R1s[threadIdx.x] = fmaxf(0.0f, St - K)/n;
        R2s[threadIdx.x] = vt/n;

        __syncthreads(); // wait for all threads to finish the computation

        i = blockDim.x/2;
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

    }


} // namespace nmch::methods::kernels

namespace nmch::methods
{

    template <typename rnd_state>
    NMCH_EM_K1<rnd_state>::NMCH_EM_K1(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {
        // each thread will have its own state
        state_numbers = NTPB * NB;
    };

    template <typename rnd_state>
    void NMCH_EM_K1<rnd_state>::init_curand_state()
    {
	    nmch::random::init_curand_state_k<<<this->NB, this->NTPB>>>(states);
    };

    template <typename rnd_state>
    void NMCH_EM_K1<rnd_state>::finalize()
    {
        cudaFree(sum);
        cudaFree(states);
    };

    template <typename rnd_state>
    void NMCH_EM_K1<rnd_state>::print_stats()
    {   
        //call the print_stats of the base class
        NMCH<rnd_state>::print_stats();
        printf("The estimated price is equal to %f\n", this->strike_price);
        printf("The estimated volatility is equal to %f\n", this->volatility);
        printf("error associated to a confidence interval of 95%% = %f\n",
            1.96 * sqrt((double)(1.0f / (this->state_numbers - 1)) * (this->state_numbers*this->volatility - 
            (this->strike_price * this->strike_price)))/sqrt((double)this->state_numbers));
        printf("The true price %f\n", this->S_0 * nmch::utils::NP((this->r + 0.5 * this->sigma * this->sigma)/this->sigma) -
                                        this->K * expf(-this->r) * nmch::utils::NP((this->r - 0.5 * this->sigma * this->sigma) /
                                        this->sigma));
        printf("Execution time %f ms\n", Tim);
    }

    // definition of the base class to avoid compilation errors
    template class NMCH_EM_K1<curandStateXORWOW_t>;
    template class NMCH_EM_K1<curandStateMRG32k3a_t>;
    template class NMCH_EM_K1<curandStatePhilox4_32_10_t>;
    
} // NMCH_EM_K1


namespace nmch::methods
{

    template <typename rnd_state>
    NMCH_EM_K1_MM<rnd_state>::NMCH_EM_K1_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_EM_K1<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void NMCH_EM_K1_MM<rnd_state>::init()
    {
        // one accumulator for the price and one for the variance
        cudaMallocManaged(&(this->sum), 2 * sizeof(float));
        cudaMemset(this->sum, 0, 2 * sizeof(float));
        cudaMallocManaged(&(this->states), this->state_numbers * sizeof(rnd_state));
        this->init_curand_state();
    };

    template <typename rnd_state>
    void
    NMCH_EM_K1_MM<rnd_state>::compute()
    {
        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

        kernels::EM_k1<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
                this->r, this->k, this->rho, this->theta, this->sigma, this->dt, this->K, this->N, this->states,
                this->sum, this->state_numbers);

        cudaDeviceSynchronize(); // we have to synchronize the device since we remove the memcopy

        cudaEventRecord(stop, 0);			// GPU timer instructions
        cudaEventSynchronize(stop);			// GPU timer instructions
        cudaEventElapsedTime(&(this->Tim),			// GPU timer instructions
            start, stop);					// GPU timer instructions
        cudaEventDestroy(start);			// GPU timer instructions
        cudaEventDestroy(stop);				// GPU timer instructions

        //cudaMemcpy(&(this->result), this->sum, sizeof(float), cudaMemcpyDeviceToHost);

        this->strike_price = this->sum[0];
        this->volatility = this->sum[1];
    };

    // definition of the base class to avoid compilation errors
    template class NMCH_EM_K1_MM<curandStateXORWOW_t>;
    template class NMCH_EM_K1_MM<curandStateMRG32k3a_t>;
    template class NMCH_EM_K1_MM<curandStatePhilox4_32_10_t>;
    
} // NMCH_EM_K1_MM

/* 
namespace nmch::methods
{
    template <typename rnd_state>
    NMCH_FE_K2_PgM<rnd_state>::NMCH_FE_K2_PgM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K2<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void NMCH_FE_K2_PgM<rnd_state>::init()
    {
        // one accumulator for the price and one for the variance
        cudaMalloc(&(this->sum), 2 * sizeof(float));
        cudaMemset(this->sum, 0, 2 * sizeof(float));
        cudaMalloc(&(this->states), this->state_numbers * sizeof(rnd_state));
        this->init_curand_state();
    };

    template <typename rnd_state>
    void
    NMCH_FE_K2_PgM<rnd_state>::compute()
    {   
        float result[2];

        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

        kernels::MC_k2<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
                this->r, this->k, this->rho, this->theta, this->sigma, this->dt, this->K, this->N, this->states,
                this->sum, this->state_numbers);

        // no need to synchronize the device since we are using the memcopy after.

        cudaEventRecord(stop, 0);			// GPU timer instructions
        cudaEventSynchronize(stop);			// GPU timer instructions
        cudaEventElapsedTime(&(this->Tim),			// GPU timer instructions
            start, stop);					// GPU timer instructions
        cudaEventDestroy(start);			// GPU timer instructions
        cudaEventDestroy(stop);				// GPU timer instructions

        cudaMemcpy(&result, this->sum, 2*sizeof(float), cudaMemcpyDeviceToHost);

        this->strike_price = result[0];
        this->volatility = result[1];
    };

    template class NMCH_FE_K2_PgM<curandStateXORWOW_t>;
    template class NMCH_FE_K2_PgM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K2_PgM<curandStatePhilox4_32_10_t>;
} //NMCH_FE_K2_PgM

namespace nmch::methods
{

    template <typename rnd_state>
    NMCH_FE_K2_PiM<rnd_state>::NMCH_FE_K2_PiM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K2<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void 
    NMCH_FE_K2_PiM<rnd_state>::init()
    {
        //pinning the memory on the host
        //cudaHostAlloc(&result, 2*sizeof(float), cudaHostAllocDefault);
        cudaMallocHost(&result, 2*sizeof(float));

        // one accumulator for the price and one for the variance
        cudaMalloc(&(this->sum), 2 * sizeof(float));
        cudaMemset(this->sum, 0, 2 * sizeof(float));
        cudaMalloc(&(this->states), this->state_numbers * sizeof(rnd_state));
        this->init_curand_state();
    };

    template <typename rnd_state>
    void 
    NMCH_FE_K2_PiM<rnd_state>::finalize()
    {
        cudaFreeHost(result);
        NMCH_FE_K2<rnd_state>::finalize();
    };

    template <typename rnd_state>
    void
    NMCH_FE_K2_PiM<rnd_state>::compute()
    {   
        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

        kernels::MC_k2<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
                this->r, this->k, this->rho, this->theta, this->sigma, this->dt, this->K, this->N, this->states,
                this->sum, this->state_numbers);

        //cudaDeviceSynchronize(); //we are using the memcopy after.

        cudaEventRecord(stop, 0);			// GPU timer instructions
        cudaEventSynchronize(stop);			// GPU timer instructions
        cudaEventElapsedTime(&(this->Tim),			// GPU timer instructions
            start, stop);					// GPU timer instructions
        cudaEventDestroy(start);			// GPU timer instructions
        cudaEventDestroy(stop);				// GPU timer instructions

        testCUDA(cudaMemcpy(result, this->sum, 2*sizeof(float), cudaMemcpyDeviceToHost));

        this->strike_price = result[0];
        this->volatility = result[1];
    };

    template class NMCH_FE_K2_PiM<curandStateXORWOW_t>;
    template class NMCH_FE_K2_PiM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K2_PiM<curandStatePhilox4_32_10_t>;
}
 */
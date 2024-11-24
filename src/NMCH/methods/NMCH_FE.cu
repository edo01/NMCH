#include "NMCH/methods/NMCH_FE.hpp"

#define testCUDA(error) (nmch::utils::cuda::checkCUDA(error, __FILE__ , __LINE__))


namespace nmch::methods::kernels{

    template <typename rnd_state>
    __global__ void MC_k2(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, rnd_state* state, float* sum, int n)
    {

        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        rnd_state localState = state[idx]; // in this way we avoid two different series to be the same
        float2 G1, G2;
        float S = S_0;
        float V = v_0;
        extern __shared__ float A[]; // dynamically allocated in the kernel call
        float *R1s, * R2s; 
        R1s = A;
        R2s = R1s + blockDim.x;
        int i;

        for(i = 0; i<N; i++)
        {
            G1 = curand_normal2(&localState);
            G2 = curand_normal2(&localState);

            S = S + r * S * dt + sqrtf(V)*S*sqrtf(dt)*(rho*G1.x+sqrtf(1-rho*rho)*G2.x);
            V = V + k*(theta - V)*dt + sigma*sqrtf(V)*sqrtf(dt)*G1.x;
            V = abs(V);            
        }

        R1s[threadIdx.x] = fmaxf(0.0f, S - K)/n;
        R2s[threadIdx.x] = V/n;

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

        // if am doing only one montecarlo simulation
        // I haeve to begin again the sequence
        // state[idx] = localState;
    };

} // namespace nmch::methods::kernels

namespace nmch::methods
{

    template <typename rnd_state>
    NMCH_FE_K2<rnd_state>::NMCH_FE_K2(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {
        // each thread will have its own state
        state_numbers = NTPB * NB;
    };

    template <typename rnd_state>
    void NMCH_FE_K2<rnd_state>::init_curand_state()
    {
	    nmch::random::init_curand_state_k<<<this->NB, this->NTPB>>>(states);
    };

    template <typename rnd_state>
    void NMCH_FE_K2<rnd_state>::finalize()
    {
        cudaFree(sum);
        cudaFree(states);
    };

    template <typename rnd_state>
    void NMCH_FE_K2<rnd_state>::print_stats()
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
    template class NMCH_FE_K2<curandStateXORWOW_t>;
    template class NMCH_FE_K2<curandStateMRG32k3a_t>;
    template class NMCH_FE_K2<curandStatePhilox4_32_10_t>;
    
} // NMCH_FE_K2


namespace nmch::methods
{

    template <typename rnd_state>
    NMCH_FE_K2_MM<rnd_state>::NMCH_FE_K2_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K2<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void NMCH_FE_K2_MM<rnd_state>::init()
    {
        // one accumulator for the price and one for the variance
        cudaMallocManaged(&(this->sum), 2 * sizeof(float));
        cudaMemset(this->sum, 0, 2 * sizeof(float));
        cudaMallocManaged(&(this->states), this->state_numbers * sizeof(rnd_state));
        this->init_curand_state();
    };

    template <typename rnd_state>
    void
    NMCH_FE_K2_MM<rnd_state>::compute()
    {
        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

        kernels::MC_k2<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
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
    template class NMCH_FE_K2_MM<curandStateXORWOW_t>;
    template class NMCH_FE_K2_MM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K2_MM<curandStatePhilox4_32_10_t>;
    
} // NMCH_FE_K2_MM


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

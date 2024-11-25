#include "NMCH/methods/NMCH_FE.hpp"

#define testCUDA(error) (nmch::utils::cuda::checkCUDA(error, __FILE__ , __LINE__))


namespace nmch::methods::kernels{

    template <typename rnd_state>
    __global__ void FE_k1(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, rnd_state* state, float* sum, int n)
    {

        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        extern __shared__ float A[]; // dynamically allocated in the kernel call
        // pointers to the shared memory
        float *SR, *VR; 
        SR = A; // stock price reduction 
        VR = SR + blockDim.x; // variance reduction

        // initialize the random state
        rnd_state localState = state[idx]; 

        float2 G;
        float St = S_0;
        float Vt = v_0;
        int i;
        const float sqrt_dt = sqrtf(dt);
        const float sqrt_rho = sqrtf(1-rho*rho);

        /*  ###################################
                    FORWARD EULER
            ################################### */
        for(i = 0; i<N; i++)
        {
            G = curand_normal2(&localState); // returns two normally distributed numbers

            St = St + r * St * dt + sqrtf(Vt)*St*sqrt_dt*(rho*G.x+sqrt_rho*G.y); // maybe sqrtf(Vt) also??
            Vt = Vt + k*(theta - Vt)*dt + sigma*sqrtf(Vt)*sqrt_dt*G.x;
            Vt = abs(Vt);        
            if(blockIdx.x == 0 && threadIdx.x == 0)
            {
                printf("St = %f, Vt = %f\n", St, Vt);
            }    
        }
        // St = S1, Vt = V1

        /**
         * ###################################
                        REDUCTION
            ###################################
         */
        SR[threadIdx.x] = fmaxf(0.0f, St - K)/n;
        VR[threadIdx.x] = Vt/n;

        __syncthreads(); // wait for all threads to finish the computation

        i = blockDim.x/2;
        while(i != 0)
        {
            if(threadIdx.x < i)
            {
                SR[threadIdx.x] += SR[threadIdx.x + i];
                VR[threadIdx.x] += VR[threadIdx.x + i];
            }
            __syncthreads(); // wait for all threads to finish the computation
            i /= 2;
        }

        if(threadIdx.x == 0)
        {
            atomicAdd(sum, SR[0]);
            atomicAdd(sum +1, VR[0]);
        }

        // if am doing only one montecarlo simulation
        // I haeve to begin again the sequence
        // state[idx] = localState;
    };

} // namespace nmch::methods::kernels

namespace nmch::methods
{

    template <typename rnd_state>
    NMCH_FE_K1<rnd_state>::NMCH_FE_K1(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {
        // each thread will have its own state
        state_numbers = NTPB * NB;
    };

    template <typename rnd_state>
    void NMCH_FE_K1<rnd_state>::init_curand_state()
    {
	    nmch::random::init_curand_state_k<<<this->NB, this->NTPB>>>(states);
    };

    template <typename rnd_state>
    void NMCH_FE_K1<rnd_state>::finalize()
    {
        cudaFree(sum);
        cudaFree(states);
    };

    template <typename rnd_state>
    void NMCH_FE_K1<rnd_state>::print_stats()
    {   
        //call the print_stats of the base class
        NMCH<rnd_state>::print_stats();
        printf("The estimated price is equal to %f\n", this->strike_price);
        printf("The estimated variance is equal to %f\n", this->variance);
        printf("error associated to a confidence interval of 95%% = %f\n",
            1.96 * sqrt((double)(1.0f / (this->state_numbers - 1)) * (this->state_numbers*this->variance - 
            (this->strike_price * this->strike_price)))/sqrt((double)this->state_numbers));
        printf("The true price %f\n", this->S_0 * nmch::utils::NP((this->r + 0.5 * this->sigma * this->sigma)/this->sigma) -
                                        this->K * expf(-this->r) * nmch::utils::NP((this->r - 0.5 * this->sigma * this->sigma) /
                                        this->sigma));
        printf("Execution time %f ms\n", Tim);
    }

    // definition of the base class to avoid compilation errors
    template class NMCH_FE_K1<curandStateXORWOW_t>;
    template class NMCH_FE_K1<curandStateMRG32k3a_t>;
    template class NMCH_FE_K1<curandStatePhilox4_32_10_t>;
    
} // NMCH_FE_K1


namespace nmch::methods
{

    template <typename rnd_state>
    NMCH_FE_K1_MM<rnd_state>::NMCH_FE_K1_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K1<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void NMCH_FE_K1_MM<rnd_state>::init()
    {
        // one accumulator for the price and one for the variance
        cudaMallocManaged(&(this->sum), 2 * sizeof(float));
        cudaMemset(this->sum, 0, 2 * sizeof(float));
        cudaMallocManaged(&(this->states), this->state_numbers * sizeof(rnd_state));
        this->init_curand_state();
    };

    template <typename rnd_state>
    void
    NMCH_FE_K1_MM<rnd_state>::compute()
    {
        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

        kernels::FE_k1<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
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
        this->variance = this->sum[1];
    };

    // definition of the base class to avoid compilation errors
    template class NMCH_FE_K1_MM<curandStateXORWOW_t>;
    template class NMCH_FE_K1_MM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K1_MM<curandStatePhilox4_32_10_t>;
    
} // NMCH_FE_K1_MM


namespace nmch::methods
{
    template <typename rnd_state>
    NMCH_FE_K1_PgM<rnd_state>::NMCH_FE_K1_PgM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K1<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void NMCH_FE_K1_PgM<rnd_state>::init()
    {
        // one accumulator for the price and one for the variance
        cudaMalloc(&(this->sum), 2 * sizeof(float));
        cudaMemset(this->sum, 0, 2 * sizeof(float));
        cudaMalloc(&(this->states), this->state_numbers * sizeof(rnd_state));
        this->init_curand_state();
    };

    template <typename rnd_state>
    void
    NMCH_FE_K1_PgM<rnd_state>::compute()
    {   
        float result[2];

        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

        kernels::FE_k1<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
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
        this->variance = result[1];
    };

    template class NMCH_FE_K1_PgM<curandStateXORWOW_t>;
    template class NMCH_FE_K1_PgM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K1_PgM<curandStatePhilox4_32_10_t>;
} //NMCH_FE_K1_PgM

namespace nmch::methods
{

    template <typename rnd_state>
    NMCH_FE_K1_PiM<rnd_state>::NMCH_FE_K1_PiM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K1<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void 
    NMCH_FE_K1_PiM<rnd_state>::init()
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
    NMCH_FE_K1_PiM<rnd_state>::finalize()
    {
        cudaFreeHost(result);
        NMCH_FE_K1<rnd_state>::finalize();
    };

    template <typename rnd_state>
    void
    NMCH_FE_K1_PiM<rnd_state>::compute()
    {   
        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

        kernels::FE_k1<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
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
        this->variance = result[1];
    };

    template class NMCH_FE_K1_PiM<curandStateXORWOW_t>;
    template class NMCH_FE_K1_PiM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K1_PiM<curandStatePhilox4_32_10_t>;
}

#include "NMCH/methods/NMCH_FE.hpp"

#define testCUDA(error) (nmch::utils::cuda::checkCUDA(error, __FILE__ , __LINE__))


namespace nmch::methods::kernels{


    /**
     * K1  features:
     * - It uses the Forward Euler method
     * - It reduces the results of the simulation on the device
     * - It saves the state of the random number generator   
     * - It uses aggregated calls to the random number generator to reduce the overhead of the calls
     */
    template <typename rnd_state>
    __global__ void FE_k1(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, rnd_state* state, float* sum, int n)
    {

        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        extern __shared__ float A[]; // dynamically allocated in the kernel call
        // pointers to the shared memory
        float *SR, *VR; 
        SR = A; // stock price reduction 
        VR = SR + blockDim.x; // price_squared reduction

        // initialize the random state
        rnd_state localState = state[tid]; 

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

            St = St + r * St * dt + sqrtf(Vt)*St*sqrt_dt*(rho*G.x+sqrt_rho*G.y); 
            Vt = Vt + k*(theta - Vt)*dt + sigma*sqrtf(Vt)*sqrt_dt*G.x;
            Vt = abs(Vt);
        }
        // St = S1, Vt = V1

        /**
         * ###################################
                        REDUCTION
            ###################################
         */
        SR[threadIdx.x] = fmaxf(0.0f, St - K)/n;
        //VR[threadIdx.x] = Vt/n;
        VR[threadIdx.x] = SR[threadIdx.x]*SR[threadIdx.x]/n;

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

        // when exploring the parameters we need to store the current state in the global memory
        state[tid] = localState;
    };
 
    // Perform warp-level reduction
    __inline__ __device__ float warpReduceSum(float val) 
    {
        for (int offset = 16; offset > 0; offset /= 2) {
            // 0xFFFFFFFF each warp contribute
            // val is the register to be shifted
            // offset is the distance to shift
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        return val;
    }

    // Perform block-level reduction of the warp reduced values
    __inline__ __device__ float blockReduceSum(float val) 
    {
        /*if the compute capability is lower than 7.0, we are allocating more shared memory than required 
        because the maximum number of threads per block is 512 instead of 1024*/
        static __shared__ float shared[32]; // Shared memory for one value per warp
        int lane = threadIdx.x % 32;        // Lane index within the warp
        int warpId = threadIdx.x / 32;      // Warp index within the block

        // Perform warp-level reduction
        val = warpReduceSum(val); // WE ARE ASSUMING A NUMBER OF THREADS PER BLOCK WHICH IS A MULTIPLE OF THE WARP(not a strong assumption)

        // Write the reduced value of each warp to shared memory (only the first thread of each warp)
        if (lane == 0) shared[warpId] = val; 

        __syncthreads();

        // Let the first warp reduce all warp results
        /*
            At this point some shared memory may be not used
            This may be caused from 2 reasons:
            - we are using a compute capability lower than 7.0
            - the number of threads per block are not the maximum possible
            In this case we will not use all the first warp but just
            the first blockDim.x/32 threads
        */
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0; 
        if (warpId == 0) val = warpReduceSum(val); // no divergence here

        return val;
    }

    /**
     * K1 uses 2 * number of threads per block floats of shared memory, while this new version is using only
     * 32 floats of shared memory per block.
     *
     * K2 version of the kernel has the same features of the K1 version but it uses warp shuffle instructions for 
     * the reduction.
     */
    template <typename rnd_state>
    __global__ void FE_k2(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, rnd_state* state, float* sum, int n)
    {

        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        // initialize the random state
        rnd_state localState = state[tid]; 

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
        }
        // St = S1, Vt = V1

        /*  ###################################
                    REDUCTION
        ################################### */

        // Perform block-level reduction
        float partialS, partialS2;
        partialS = fmaxf(0.0f, St - K);
        partialS2 = partialS*partialS;
        partialS = blockReduceSum(partialS/n);
        partialS2 = blockReduceSum(partialS2/n);

        // Use atomicAdd to accumulate the partial sum of the blocks
        if (threadIdx.x == 0){
            atomicAdd(sum, partialS);
            atomicAdd(sum + 1, partialS2);
        }

        // when exploring the parameters we need to store the current state in the global memory
        state[tid] = localState;

    };

    /**
     * This kernel only works with the curandStatePhilox4_32_10_t random state
     * we call the curand_normal4 function to generate 4 normally distributed numbers
     */
    __global__ void FE_k2_philox(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, curandStatePhilox4_32_10_t* state, float* sum, int n)
    {

        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        // initialize the random state
        curandStatePhilox4_32_10_t localState = state[tid]; 

        float4 G;
        float St = S_0;
        float Vt = v_0;
        int i;
        const float sqrt_dt = sqrtf(dt);
        const float sqrt_rho = sqrtf(1-rho*rho);

        /*  ###################################
                    FORWARD EULER
            ################################### */
        
        for(i = 0; i<N; i+=2)
        {
            G = curand_normal4(&localState); // returns four normally distributed numbers

            St = St + r * St * dt + sqrtf(Vt)*St*sqrt_dt*(rho*G.x+sqrt_rho*G.y);
            Vt = Vt + k*(theta - Vt)*dt + sigma*sqrtf(Vt)*sqrt_dt*G.x;
            Vt = abs(Vt);

            St = St + r * St * dt + sqrtf(Vt)*St*sqrt_dt*(rho*G.z+sqrt_rho*G.w);
            Vt = Vt + k*(theta - Vt)*dt + sigma*sqrtf(Vt)*sqrt_dt*G.z;
            Vt = abs(Vt);
        }
        // St = S1, Vt = V1

        /*  ###################################
                    REDUCTION
        ################################### */

        // Perform block-level reduction
        float partialS, partialS2;
        partialS = fmaxf(0.0f, St - K);
        partialS2 = partialS*partialS;
        partialS = blockReduceSum(partialS/n);
        partialS2 = blockReduceSum(partialS2/n);

        // Use atomicAdd to accumulate the partial sum of the blocks
        if (threadIdx.x == 0){
            atomicAdd(sum, partialS);
            atomicAdd(sum + 1, partialS2);
        }

        // during the exploaration we need to store the current state in the global memory
        state[tid] = localState;
    };

    template <typename rnd_state>
    __global__ void FE_k3(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, rnd_state* state, float* sum, int n)
    {
        // For GPUs with compute capability 8.6 maximum shared memory per thread block is 99 KB.
        // In the worst case it is 64 Bytes per thread * 512 = 32 KB which 
        /**
         * We can't use more than 512 threads otherwise we don't have enough shared memory
         *
         */
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ rnd_state shared_states[512];

        // Save the states in the shared memory
        shared_states[threadIdx.x] = state[tid]; 

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
            G = curand_normal2(&shared_states[threadIdx.x]); // returns two normally distributed numbers

            St = St + r * St * dt + sqrtf(Vt)*St*sqrt_dt*(rho*G.x+sqrt_rho*G.y); // maybe sqrtf(Vt) also??
            Vt = Vt + k*(theta - Vt)*dt + sigma*sqrtf(Vt)*sqrt_dt*G.x;
            Vt = abs(Vt);
        }
        // St = S1, Vt = V1

        /*  ###################################
                    REDUCTION
        ################################### */

        // Perform block-level reduction
        float partialS, partialS2;
        partialS = fmaxf(0.0f, St - K);
        partialS2 = partialS*partialS;
        partialS = blockReduceSum(partialS/n);
        partialS2 = blockReduceSum(partialS2/n);

        // Use atomicAdd to accumulate the partial sum of the blocks
        if (threadIdx.x == 0){
            atomicAdd(sum, partialS);
            atomicAdd(sum + 1, partialS2);
        }

        // when exploring the parameters we need to store the current state in the global memory
        state[tid] = shared_states[threadIdx.x];
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
    void NMCH_FE_K1<rnd_state>::init_curand_state(unsigned long long seed)
    {
	    nmch::random::init_curand_state_k<<<this->NB, this->NTPB>>>(states, seed);
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
        float real_price = this->S_0 * nmch::utils::NP((this->r + 0.5 * this->sigma * this->sigma)/this->sigma) -
                                        this->K * expf(-this->r) * nmch::utils::NP((this->r - 0.5 * this->sigma * this->sigma) /
                                        this->sigma);
        //call the print_stats of the base class
        NMCH<rnd_state>::print_stats();
        printf("METHOD: FORWARD-EULER\n");
        printf("The estimated price E[X] is equal to %f\n", this->strike_price);
        printf("The estimated E[X^2] is equal to %f\n", this->price_squared);
        printf("The true price %f\n", real_price);
        printf("error associated to a confidence interval of 95%% = %f\n",
            1.96 * sqrt((double)(1.0f / (this->state_numbers - 1)) * (this->state_numbers*this->price_squared - 
            (this->strike_price * this->strike_price)))/sqrt((double)this->state_numbers));
        printf("Execution time %f ms\n", Tim_exec);
        printf("Initialization time %f ms\n", Tim_init);
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
    void NMCH_FE_K1_MM<rnd_state>::init(unsigned long long seed)
    {
        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);			
        
        // one accumulator for the price and one for the price_squared
        cudaMallocManaged(&(this->sum), 2 * sizeof(float));
        cudaMemset(this->sum, 0, 2 * sizeof(float));
        cudaMalloc(&(this->states), this->state_numbers * sizeof(rnd_state));
        this->init_curand_state(seed);
        
        cudaEventRecord(stop, 0);			
        cudaEventSynchronize(stop);			
        cudaEventElapsedTime(&(this->Tim_init), start, stop);					
        cudaEventDestroy(start);			
        cudaEventDestroy(stop);		
    };

    template <typename rnd_state>
    void
    NMCH_FE_K1_MM<rnd_state>::compute()
    {
        // memset sum to 0 for multiple runs
        cudaMemset(this->sum, 0, 2 * sizeof(float));

        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);			

        kernels::FE_k1<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
                this->r, this->k, this->rho, this->theta, this->sigma, this->dt, this->K, this->N, this->states,
                this->sum, this->state_numbers);

        cudaDeviceSynchronize(); // we have to synchronize the device since we remove the memcopy

        cudaEventRecord(stop, 0);			// GPU timer instructions
        cudaEventSynchronize(stop);			// GPU timer instructions
        cudaEventElapsedTime(&(this->Tim_exec),			// GPU timer instructions
            start, stop);					// GPU timer instructions
        cudaEventDestroy(start);			// GPU timer instructions
        cudaEventDestroy(stop);				// GPU timer instructions

        //cudaMemcpy(&(this->result), this->sum, sizeof(float), cudaMemcpyDeviceToHost);

        this->strike_price = this->sum[0];
        this->price_squared = this->sum[1];
    };

    // definition of the base class to avoid compilation errors
    template class NMCH_FE_K1_MM<curandStateXORWOW_t>;
    template class NMCH_FE_K1_MM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K1_MM<curandStatePhilox4_32_10_t>;
    
} // NMCH_FE_K1_MM

namespace nmch::methods
{
    template <typename rnd_state>
    NMCH_FE_K2_MM<rnd_state>::NMCH_FE_K2_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K1_MM<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void
    NMCH_FE_K2_MM<rnd_state>::compute()
    {
        // memset sum to 0 for multiple runs
        cudaMemset(this->sum, 0, 2 * sizeof(float));

        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);			

        kernels::FE_k2<<<this->NB, this->NTPB>>>(this->S_0, this->v_0,
                this->r, this->k, this->rho, this->theta, this->sigma, this->dt, this->K, this->N, this->states,
                this->sum, this->state_numbers);
        

        cudaDeviceSynchronize(); // we have to synchronize the device since we remove the memcopy

        cudaEventRecord(stop, 0);			// GPU timer instructions
        cudaEventSynchronize(stop);			// GPU timer instructions
        cudaEventElapsedTime(&(this->Tim_exec),			// GPU timer instructions
            start, stop);					// GPU timer instructions
        cudaEventDestroy(start);			// GPU timer instructions
        cudaEventDestroy(stop);				// GPU timer instructions

        //cudaMemcpy(&(this->result), this->sum, sizeof(float), cudaMemcpyDeviceToHost);

        this->strike_price = this->sum[0];
        this->price_squared = this->sum[1];
    };

    // definition of the base class to avoid compilation errors
    template class NMCH_FE_K2_MM<curandStateXORWOW_t>;
    template class NMCH_FE_K2_MM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K2_MM<curandStatePhilox4_32_10_t>;
} // NMCH_FE_K2_MM

namespace nmch::methods
{
    NMCH_FE_K2_PHILOX_MM::NMCH_FE_K2_PHILOX_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K1_MM<curandStatePhilox4_32_10_t>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    void
    NMCH_FE_K2_PHILOX_MM::compute()
    {
        // memset sum to 0 for multiple runs
        cudaMemset(this->sum, 0, 2 * sizeof(float));

        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);			

        kernels::FE_k2_philox<<<this->NB, this->NTPB>>>(this->S_0, this->v_0,
                this->r, this->k, this->rho, this->theta, this->sigma, this->dt, this->K, this->N, this->states,
                this->sum, this->state_numbers);
        
        cudaDeviceSynchronize(); // we have to synchronize the device since we removed the memcopy

        cudaEventRecord(stop, 0);			// GPU timer instructions
        cudaEventSynchronize(stop);			// GPU timer instructions
        cudaEventElapsedTime(&(this->Tim_exec),			// GPU timer instructions
            start, stop);					// GPU timer instructions
        cudaEventDestroy(start);			// GPU timer instructions
        cudaEventDestroy(stop);				// GPU timer instructions

        //cudaMemcpy(&(this->result), this->sum, sizeof(float), cudaMemcpyDeviceToHost);

        this->strike_price = this->sum[0];
        this->price_squared = this->sum[1];
    };

} // NMCH_FE_K2_PHILOX_MM

namespace nmch::methods
{
    template <typename rnd_state>
    NMCH_FE_K3_MM<rnd_state>::NMCH_FE_K3_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K2_MM<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void
    NMCH_FE_K3_MM<rnd_state>::compute()
    {
        // memset sum to 0 for multiple runs
        cudaMemset(this->sum, 0, 2 * sizeof(float));

        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);			

        kernels::FE_k3<<<this->NB, this->NTPB>>>(this->S_0, this->v_0,
                this->r, this->k, this->rho, this->theta, this->sigma, this->dt, this->K, this->N, this->states,
                this->sum, this->state_numbers);
        

        cudaDeviceSynchronize(); // we have to synchronize the device since we remove the memcopy

        cudaEventRecord(stop, 0);			// GPU timer instructions
        cudaEventSynchronize(stop);			// GPU timer instructions
        cudaEventElapsedTime(&(this->Tim_exec),			// GPU timer instructions
            start, stop);					// GPU timer instructions
        cudaEventDestroy(start);			// GPU timer instructions
        cudaEventDestroy(stop);				// GPU timer instructions

        //cudaMemcpy(&(this->result), this->sum, sizeof(float), cudaMemcpyDeviceToHost);

        this->strike_price = this->sum[0];
        this->price_squared = this->sum[1];
    };

    // definition of the base class to avoid compilation errors
    template class NMCH_FE_K3_MM<curandStateXORWOW_t>;
    template class NMCH_FE_K3_MM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K3_MM<curandStatePhilox4_32_10_t>;
} // NMCH_FE_K3_MM

namespace nmch::methods
{
    template <typename rnd_state>
    NMCH_FE_K1_PgM<rnd_state>::NMCH_FE_K1_PgM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_FE_K1<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};

    template <typename rnd_state>
    void NMCH_FE_K1_PgM<rnd_state>::init(unsigned long long seed)
    {
        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);	

        // one accumulator for the price and one for the price_squared
        cudaMalloc(&(this->sum), 2 * sizeof(float));
        cudaMemset(this->sum, 0, 2 * sizeof(float));
        cudaMalloc(&(this->states), this->state_numbers * sizeof(rnd_state));
        this->init_curand_state(seed);

        cudaEventRecord(stop, 0);			
        cudaEventSynchronize(stop);			
        cudaEventElapsedTime(&(this->Tim_init), start, stop);					
        cudaEventDestroy(start);			
        cudaEventDestroy(stop);		
    };

    template <typename rnd_state>
    void
    NMCH_FE_K1_PgM<rnd_state>::compute()
    {   
        float result[2];

        // memset sum to 0 for multiple runs
        cudaMemset(this->sum, 0, 2 * sizeof(float));

        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);			

        kernels::FE_k1<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
                this->r, this->k, this->rho, this->theta, this->sigma, this->dt, this->K, this->N, this->states,
                this->sum, this->state_numbers);

        // no need to synchronize the device since we are using the memcopy after.

        cudaEventRecord(stop, 0);			
        cudaEventSynchronize(stop);			
        cudaEventElapsedTime(&(this->Tim_exec), start, stop);		
        cudaEventDestroy(start);
        cudaEventDestroy(stop);	

        cudaMemcpy(&result, this->sum, 2*sizeof(float), cudaMemcpyDeviceToHost);

        this->strike_price = result[0];
        this->price_squared = result[1];
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
    NMCH_FE_K1_PiM<rnd_state>::init(unsigned long long seed)
    {
        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);	

        //pinning the memory on the host
        //cudaHostAlloc(&result, 2*sizeof(float), cudaHostAllocDefault);
        cudaMallocHost(&result, 2*sizeof(float));

        // one accumulator for the price and one for the price_squared
        cudaMalloc(&(this->sum), 2 * sizeof(float));
        cudaMemset(this->sum, 0, 2 * sizeof(float));
        cudaMalloc(&(this->states), this->state_numbers * sizeof(rnd_state));
        this->init_curand_state(seed);

        cudaEventRecord(stop, 0);			
        cudaEventSynchronize(stop);			
        cudaEventElapsedTime(&(this->Tim_init), start, stop);					
        cudaEventDestroy(start);			
        cudaEventDestroy(stop);		
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
        // memset sum to 0
        cudaMemset(this->sum, 0, 2 * sizeof(float));

        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);			

        kernels::FE_k1<<<this->NB, this->NTPB, 2 * this->NTPB * sizeof(float)>>>(this->S_0, this->v_0,
                this->r, this->k, this->rho, this->theta, this->sigma, this->dt, this->K, this->N, this->states,
                this->sum, this->state_numbers);

        //cudaDeviceSynchronize(); //we are using the memcopy after.

        cudaEventRecord(stop, 0);			
        cudaEventSynchronize(stop);			
        cudaEventElapsedTime(&(this->Tim_exec), start, stop);					
        cudaEventDestroy(start);			
        cudaEventDestroy(stop);				

        testCUDA(cudaMemcpy(result, this->sum, 2*sizeof(float), cudaMemcpyDeviceToHost));

        this->strike_price = result[0];
        this->price_squared = result[1];
    };

    template class NMCH_FE_K1_PiM<curandStateXORWOW_t>;
    template class NMCH_FE_K1_PiM<curandStateMRG32k3a_t>;
    template class NMCH_FE_K1_PiM<curandStatePhilox4_32_10_t>;
} //NMCH_FE_K1_PiM

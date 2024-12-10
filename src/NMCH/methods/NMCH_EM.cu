/**
 * Ideas for kernel optimization: 
 * - separate the simulation and the reduction and optimize the reduction as much as possible and then maybe merge the two kernels
 * - use the hardware exp instruction and the hardware sqrt instruction
 * - batch random number generation: USING LESS RANDOM NUMBER STATES??
 https://forums.developer.nvidia.com/t/question-about-optimal-curand-use/37752
 */
#include "NMCH/methods/NMCH_EM.hpp"

#define testCUDA(error) (nmch::utils::cuda::checkCUDA(error, __FILE__ , __LINE__))

#define SIZE 32

namespace nmch::methods::kernels{
    
    template <typename rnd_state>
    __inline__ __device__
    float gamma_distribution(rnd_state* state, float alpha) 
    {
        float d, c, x, v, u, x2;
        
        // if alpha < 1: // we set alpha = alpha + 1 and we use the fact that gamma_alpha = gamma_{alpha + 1} * U^{1/alpha}
        //
        // 1. setup d=a-1/3, c=1/sqrt(9d)
        // 2. generate v = (1 + cX)^3 with x ~ N(0,1)
        // 3. repeat until v > 0
        // 4. generate U ~ U(0,1)
        // 5. if U < 1 - 0.0331x^4 return d*v (or d*v*U^(1/a) if a < 1)
        // 6. if log(U) < 0.5x^2 + d(1-v + log(v)) return dv (or d*v*U^(1/a) if a < 1)
        // else goto 2

        float C;

        if (alpha < 1.0f) {
            C = powf(curand_uniform(state), 1.0f / alpha);  // U^(1/alpha) for alpha < 1
            alpha += 1.0f;  // Increment alpha
        } else {
            C = 1.0f;  // No scaling for alpha >= 1
        }

        // step 1
        d = alpha - 1.0f / 3.0f;
        c = 1.0f / sqrtf(9.0f * d);

        while (true) {
            // step 2
            do{ x = curand_normal(state); v = 1.0f + c * x; }while (v <= 0.0f);
            v = v * v * v;
            // step 3
            u = curand_uniform(state);
            // step 5 and 6
            x2 = x * x;  // Precompute x^2
            if (u < 1.0f - 0.0331f * x2 * x2 || 
                logf(u) < 0.5f * x2 + d * (1.0f - v + logf(v))) return d * v * C;
        }
    }


    template <typename rnd_state>
    __global__ 
    void EM_k1(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, rnd_state* state, float* sum, int n)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        extern __shared__ float A[]; // dynamically allocated shared memory
        // pointers to the shared memory
        float *SR, *VR; 
        SR = A; // stock price reduction 
        VR = SR + blockDim.x; // variance reduction

        // get the local state
        rnd_state localState = state[tid];

        int i;
        int N_p; // poisson 
        float lambda, gamma, Vt_next, m, sigma2;

        // initialization of the variance and the price
        float St = S_0;
        float Vt = v_0;
        float vI = 0.0f; // accumulated variance using the trapezoidal rule
        // initializing constansts
        /* WE CAN TRY __expf(-k * dt) instead of expf(-k * dt) 
        is more efficient because it uses the hardware exp instruction 
        but it is less precise*/

        const float exp_kdt = expf(-k * dt); //expf is very expensive to compute
        const float d = 2.0f * k * theta / (sigma * sigma);
        // this part of lambda is constant, no need to compute it at each iteration
        const float lambda_const = (2 * k * exp_kdt) / (sigma * sigma * (1 - exp_kdt)); 

        /*##############################################
         *                  SIMULATION
         *##############################################*/
        for (i = 0; i < N; ++i) { // advancing in time
            // step 1
            lambda = lambda_const * Vt; 
            N_p = curand_poisson(&localState, lambda);
            gamma = gamma_distribution(&localState, d + N_p);
            // a lot of divergence here since the gamma distribution is not equally distributed among threads
            Vt_next = (sigma * sigma * (1.0f - exp_kdt) / (2.0f * k)) * gamma;

            // step 2
            vI += (Vt + Vt_next);//*dt; // dt missing????

            // advance the variance
            Vt = Vt_next;
        }
        vI *= dt*0.5; // only done once for numerical stability
        //Vt = v1;
        //step 3 -  Assuming T = 1
        m       = (1.0f / sigma) * (Vt - v_0 - k * theta + k * vI);
        // step 4 
        m       = -0.5f * vI + rho * m;
        // assume S_0 = 1
        sigma2  = (1.0f - rho * rho) * vI;
        //St
        // what happens if we use the hardware exp instruction?
        // what happens if we change curand_normal to curand_normal2?
        St      = expf(m + sqrtf(sigma2) * curand_normal(&localState));

        /*##############################################
         *                  REDUCTION
         *##############################################*/
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
            atomicAdd(sum,      SR[0]);
            atomicAdd(sum + 1,  VR[0]);
        }

        // during the exploaration we need to store the current state in the global memory
        state[idx] = localState;
    }

    __inline__ __device__ float warpReduceSum(float val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            // 0xFFFFFFFF each warp contribute
            // val is the register to be shifted
            // offset is the distance to shift
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        return val;
    }

    // Perform block-level reduction of the warp reduced values
    __inline__ __device__ float blockReduceSum(float val) {
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

    template <typename rnd_state>
    __global__ 
    void EM_k2(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, rnd_state* state, float* sum, int n)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // get the local state
        rnd_state localState = state[tid];

        int i;
        int N_p; // poisson 
        float lambda, gamma, Vt_next, m, sigma2;

        // initialization of the variance and the price
        float St = S_0;
        float Vt = v_0;
        float vI = 0.0f; // accumulated variance using the trapezoidal rule
        // initializing constansts
        /* WE CAN TRY __expf(-k * dt) instead of expf(-k * dt) 
        is more efficient because it uses the hardware exp instruction 
        but it is less precise*/

        const float exp_kdt = expf(-k * dt); //expf is very expensive to compute
        const float d = 2.0f * k * theta / (sigma * sigma);
        // this part of lambda is constant, no need to compute it at each iteration
        const float lambda_const = (2 * k * exp_kdt) / (sigma * sigma * (1 - exp_kdt)); 

        /*##############################################
         *                  SIMULATION
         *##############################################*/
        for (i = 0; i < N; ++i) { // advancing in time
            // step 1
            lambda = lambda_const * Vt; 
            N_p = curand_poisson(&localState, lambda);
            gamma = gamma_distribution(&localState, d + N_p);
            //__syncwarp();
            // a lot of divergence here since the gamma distribution may not equally distributed among threads
            Vt_next = (sigma * sigma * (1.0f - exp_kdt) / (2.0f * k)) * gamma;

            // step 2
            vI += (Vt + Vt_next);//*dt; // dt missing????

            // advance the variance
            Vt = Vt_next;
        }
        vI *= dt*0.5; // only done once for numerical stability
        //Vt = v1;
        //step 3 -  Assuming T = 1
        m       = (1.0f / sigma) * (Vt - v_0 - k * theta + k * vI);
        // step 4 
        m       = -0.5f * vI + rho * m;
        // assume S_0 = 1
        sigma2  = (1.0f - rho * rho) * vI;
        //St
        // what happens if we use the hardware exp instruction?
        // what happens if we change curand_normal to curand_normal2?
        St      = expf(m + sqrtf(sigma2) * curand_normal(&localState));

        /*##############################################
         *                  REDUCTION
         *##############################################*/

        // Perform block-level reduction
        float partialS, partialV;
        partialS = blockReduceSum(fmaxf(0.0f, St - K)/n);
        partialV = blockReduceSum(Vt/n);

        // Use atomicAdd to accumulate the partial sum of the blocks
        if (threadIdx.x == 0){
            atomicAdd(sum, partialS);
            atomicAdd(sum + 1, partialV);
        }

        // during the exploaration we need to store the current state in the global memory
        state[idx] = localState;
    }
    
    template <typename rnd_state>
    __global__ 
    void EM_k3(float S_0, float v_0, float r, float k, float rho, float theta, float sigma, float dt, 
                            float K, int N, rnd_state* state, float* sum, int n)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        // For GPUs with compute capability 8.6 maximum shared memory per thread block is 99 KB.
        // In the worst case it is 64 Bytes per thread * 512 = 32 KB which 
        /**
         * We can't use more than 512 threads otherwise we don't have enough shared memory
         *
         */
        __shared__ rnd_state shared_states[512];       

        // copy the state to the shared memory
        shared_states[threadIdx.x] = state[tid];
        __syncthreads(); // I don't think we need this synchronization

        int i;
        int N_p; // poisson 
        float lambda, gamma, Vt_next, m, sigma2;

        // initialization of the variance and the price
        float St = S_0;
        float Vt = v_0;
        float vI = 0.0f; // accumulated variance using the trapezoidal rule
        // initializing constansts
        /* WE CAN TRY __expf(-k * dt) instead of expf(-k * dt) 
        is more efficient because it uses the hardware exp instruction 
        but it is less precise*/
        const float exp_kdt = expf(-k * dt); //expf is very expensive to compute
        const float d = 2.0f * k * theta / (sigma * sigma);
        // this part of lambda is constant, no need to compute it at each iteration
        const float lambda_const = (2 * k * exp_kdt) / (sigma * sigma * (1 - exp_kdt)); 

        /*##############################################
         *                  SIMULATION
         *##############################################*/
        for (i = 0; i < N; ++i) { // advancing in time
            // step 1
            lambda = lambda_const * Vt; 
            N_p = curand_poisson(&shared_states[threadIdx.x], lambda);
            gamma = gamma_distribution(&shared_states[threadIdx.x], d + N_p);
            // a lot of divergence here since the gamma distribution is not equally distributed among threads
            Vt_next = (sigma * sigma * (1.0f - exp_kdt) / (2.0f * k)) * gamma;

            // step 2
            vI += (Vt + Vt_next);//*dt; // dt missing????

            // advance the variance
            Vt = Vt_next;
        }
        vI *= dt*0.5; // only done once for numerical stability
        //Vt = v1;
        //step 3 -  Assuming T = 1
        m       = (1.0f / sigma) * (Vt - v_0 - k * theta + k * vI);
        // step 4 
        m       = -0.5f * vI + rho * m;
        // assume S_0 = 1
        sigma2  = (1.0f - rho * rho) * vI;
        //St
        // what happens if we use the hardware exp instruction?
        // what happens if we change curand_normal to curand_normal2?
        St      = expf(m + sqrtf(sigma2) * curand_normal(&shared_states[threadIdx.x]));

        /*##############################################
         *                  REDUCTION
         *##############################################*/

        // Perform block-level reduction
        float partialS, partialV;
        partialS = blockReduceSum(fmaxf(0.0f, St - K)/n);
        partialV = blockReduceSum(Vt/n);

        // Use atomicAdd to accumulate the partial sum of the blocks
        if (threadIdx.x == 0){
            atomicAdd(sum, partialS);
            atomicAdd(sum + 1, partialV);
        }

        // during the exploaration we need to store the current state in the global memory
        state[idx] = shared_states[threadIdx.x];
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
    void NMCH_EM_K1<rnd_state>::init_curand_state(unsigned long long seed)
    {
	    nmch::random::init_curand_state_k<<<this->NB, this->NTPB>>>(states, seed);
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
        float real_price = this->S_0 * nmch::utils::NP((this->r + 0.5 * this->sigma * this->sigma)/this->sigma) -
                                        this->K * expf(-this->r) * nmch::utils::NP((this->r - 0.5 * this->sigma * this->sigma) /
                                        this->sigma);
        //call the print_stats of the base class
        NMCH<rnd_state>::print_stats();
        printf("METHOD: EXACT-METHOD\n");
        printf("The estimated price is equal to %f\n", this->strike_price);
        printf("The estimated variance is equal to %f\n", this->variance);
        printf("The true price %f\n", real_price);
        printf("Relative error committed= %f\n", abs((this->strike_price - real_price)/real_price));
        printf("error associated to a confidence interval of 95%% = %f\n",
            1.96 * sqrt((double)(1.0f / (this->state_numbers - 1)) * (this->state_numbers*this->variance - 
            (this->strike_price * this->strike_price)))/sqrt((double)this->state_numbers));
        printf("Execution time %f ms\n", Tim_exec);
        printf("Initialization time %f ms\n", Tim_init);
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
    void NMCH_EM_K1_MM<rnd_state>::init(unsigned long long seed)
    {
        
        cudaEvent_t start, stop;			
        cudaEventCreate(&start);			
        cudaEventCreate(&stop);				
        cudaEventRecord(start, 0);	

        // one accumulator for the price and one for the variance
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
        cudaEventElapsedTime(&(this->Tim_exec),			// GPU timer instructions
            start, stop);					// GPU timer instructions
        cudaEventDestroy(start);			// GPU timer instructions
        cudaEventDestroy(stop);				// GPU timer instructions

        //cudaMemcpy(&(this->result), this->sum, sizeof(float), cudaMemcpyDeviceToHost);

        this->strike_price = this->sum[0];
        this->variance = this->sum[1];
    };

    // definition of the base class to avoid compilation errors
    template class NMCH_EM_K1_MM<curandStateXORWOW_t>;
    template class NMCH_EM_K1_MM<curandStateMRG32k3a_t>;
    template class NMCH_EM_K1_MM<curandStatePhilox4_32_10_t>;
    
} // NMCH_EM_K1_MM

namespace nmch::methods
{

    template <typename rnd_state>
    NMCH_EM_K2_MM<rnd_state>::NMCH_EM_K2_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_EM_K1_MM<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};


    template <typename rnd_state>
    void
    NMCH_EM_K2_MM<rnd_state>::compute()
    {
        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

        kernels::EM_k2<<<this->NB, this->NTPB>>>(this->S_0, this->v_0,
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
        this->variance = this->sum[1];
    };

    // definition of the base class to avoid compilation errors
    template class NMCH_EM_K2_MM<curandStateXORWOW_t>;
    template class NMCH_EM_K2_MM<curandStateMRG32k3a_t>;
    template class NMCH_EM_K2_MM<curandStatePhilox4_32_10_t>;
    
} // NMCH_EM_K2_MM

namespace nmch::methods
{
    template <typename rnd_state>
    NMCH_EM_K3_MM<rnd_state>::NMCH_EM_K3_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N):
    NMCH_EM_K2_MM<rnd_state>(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N)
    {};


    template <typename rnd_state>
    void
    NMCH_EM_K3_MM<rnd_state>::compute()
    {
        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

        kernels::EM_k3<<<this->NB, this->NTPB>>>(this->S_0, this->v_0,
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
        this->variance = this->sum[1];
    };

    // definition of the base class to avoid compilation errors
    template class NMCH_EM_K3_MM<curandStateXORWOW_t>;
    template class NMCH_EM_K3_MM<curandStateMRG32k3a_t>;
    template class NMCH_EM_K3_MM<curandStatePhilox4_32_10_t>;
    
} // NMCH_EM_K3_MM
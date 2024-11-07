#include "NMCH/methods/NMCH_fw_euler.hpp"

namespace nmch::methods::kernels{

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
        int i;

        for(i = 0; i<N; i++)
        {
            G = curand_normal2(&localState);
            S *= (1 + r * dt * dt + sigma * dt * G.x);
        }

        R1s[threadIdx.x] = expf(-r * dt *dt * N) * fmaxf(0.0f, S - K)/n;
        R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x]/n;

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
    NMCH_fw_euler<rnd_state>::NMCH_fw_euler(int NTPB, int NB, float T, float S_0, float K, float sigma, float r, int N):
    NMCH(NTPB, NB, T, S_0, K, sigma, r, N)
    {};

    template <typename rnd_state>
    int
    NMCH_fw_euler<rnd_state>::compute()
    {
        // always called before the initial call to the kernel
        allocate_memory();
        init_curand_state();

        cudaEvent_t start, stop;			// GPU timer instructions
        cudaEventCreate(&start);			// GPU timer instructions
        cudaEventCreate(&stop);				// GPU timer instructions
        cudaEventRecord(start, 0);			// GPU timer instructions

	    kernels::mk_2 < << this->NB, this->NTPB, 2*this->NTPB*sizeof(float)>> > ( this->S_0, this->r, 
                this->sigma, this->dt, this->K, this->N, this->state, this->sum, this->NB*this->NTPB);
        
        //cudaDeviceSynchronize(); we are using the memcopy after.

        cudaEventRecord(stop, 0);			// GPU timer instructions
        cudaEventSynchronize(stop);			// GPU timer instructions
        cudaEventElapsedTime(&Tim,			// GPU timer instructions
            start, stop);					// GPU timer instructions
        cudaEventDestroy(start);			// GPU timer instructions
        cudaEventDestroy(stop);				// GPU timer instructions

        cudaMemcpy(&(this->result), this->sum, sizeof(float), cudaMemcpyDeviceToHost);
        free_memory();

        return this->result;
    };


    // definition of the base class to avoid compilation errors
    NMCH_fw_euler<curandStateXORWOW_t> nmch1;
    NMCH_fw_euler<curandStateMRG32k3a_t> nmch2;
    NMCH_fw_euler<curandStatePhilox4_32_10_t> nmch3;
    NMCH_fw_euler<curandStateMtgp32_t> nmch4;

} // namespace nmch::methods
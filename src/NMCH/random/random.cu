#include "NMCH/random/random.hpp"

namespace nmch::random {

    // Set the state for each thread
    template <typename rnd_state>
    __global__ void init_curand_state_k(rnd_state* state) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(0, idx, 0, &state[idx]);
    };
    

    // Explicit instantiation
    template __global__ void init_curand_state_k(curandStateXORWOW_t*);
    template __global__ void init_curand_state_k(curandStateMRG32k3a_t*);
    template __global__ void init_curand_state_k(curandStatePhilox4_32_10_t*);
    //template __global__ void init_curand_state_k(curandStateMtgp32_t*);
    

} // namespace nmch::random
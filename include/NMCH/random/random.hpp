#ifdef RANDOM_CUH
#define RANDOM_CUH

#include <curand_kernel.h>

namespace nmch::random {
        
    // Set the state for each thread
    template <typename rnd_state>
    __global__ void init_curand_state_k(rnd_state* state);

} // namespace nmch::random

#endif // RANDOM_CUH

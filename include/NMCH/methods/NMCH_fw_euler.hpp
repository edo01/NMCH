#ifndef NMCH_FW_EULER_HPP
#define NMCH_FW_EULER_HPP

#include "NMCH/methods/NMCH.hpp"
#include <curand_kernel.h>

// shoudn't be exposed
namespace nmch::methods::cudakernels
{
    // we now need n, 
    template <typename rnd_state>
    __global__ void MC_k2(float S_0, float r, float sigma, float dt, float K,
						int N, rnd_state* state, float* sum, int n);
}  // nmch::methods::cudakernels

namespace nmch::methods
{
    template <typename rnd_state>
    class NMCH_fw_euler : public NMCH<rnd_state>
    {   
        public:
            NMCH_fw_euler(int NTPB, int NB, float T, float S_0, float K, float sigma, float r, int N);
            virtual int compute() override;
            virtual ~NMCH_fw_euler() = default;
    };

} // nmch::methods

#endif // "NMCH_FW_EULER_HPP"
#ifndef NMCH_CUH
#define NMCH_CUH

#include <stdio.h>
#include "NMCH/random/random.hpp"
#include "NMCH/utils/utils.hpp"

namespace nmch::methods
{
    template <typename rnd_state>
    class NMCH {
        public:
            NMCH(int NTPB, int NB, float T, float S_0, float K, float sigma, float r, int N);
            virtual int compute() = 0; // it will be extended by the subclasses
            virtual void print_stats(); // it will be extended by the subclasses
            virtual ~NMCH() = default;

        protected:
            int NTPB;
            int NB;
            float T;
            float S_0;
            float K;
            float sigma;
            float r;
            int N;
            float dt;
            float *sum;
            float result;
            rnd_state *states; // random number generator array
            int state_numbers;  

            virtual void init_curand_state();
            virtual void allocate_memory();
            virtual void free_memory();
    };

} // namespace nmch::methods


#endif // NMCH_CUH
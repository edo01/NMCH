#ifndef NMCH_FW_EULER_HPP
#define NMCH_FW_EULER_HPP

#include <curand_kernel.h>
#include "NMCH/methods/NMCH.hpp"
#include "NMCH/utils/utils.hpp"

// shoudn't be exposed
namespace nmch::methods::cudakernels
{
    // we now need n, 
    template <typename rnd_state>
    __global__ void FE_k1(float S_0, float r, float sigma, float dt, float K,
						int N, rnd_state* state, float* sum, int n);
}  // nmch::methods::cudakernels

namespace nmch::methods
{
    template <typename rnd_state>
    class NMCH_FE_K1 : public NMCH<rnd_state>
    {   
        public:
            /**
            * @param NTPB Number of threads per block
            * @param NB Number of blocks
            * @param T Time of the maturity
            * @param S_0 Spot values
            * @param r risk-free interest rate  
            * @param k mean reversion rate of the variance
            * @param theta long-term variance
            * @param sigma volatility of the variance
            * @param N Number of time steps            
            */
            NMCH_FE_K1(int NTPB, int NB, float T, float S_0, float v_0, float r, float k,float rho, float theta, float sigma, int N);
            virtual void finalize() override;
            virtual void print_stats() override;
            virtual ~NMCH_FE_K1() = default;
        
        protected:
            /* array for performing the reduction */
            float *sum;
            /* random states of the threads */
            rnd_state *states;
            /* number of states*/
            int state_numbers;  
            /* execution time */
            float Tim_exec;
            /* initialization time */
            float Tim_init;

            /**
                Initialize the random states of the threads
            */
            virtual void init_curand_state(unsigned long long seed);
    };

    template <typename rnd_state>
    class NMCH_FE_K1_MM : public NMCH_FE_K1<rnd_state> {
        public:
            NMCH_FE_K1_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual void init(unsigned long long seed) override;
            virtual ~NMCH_FE_K1_MM() = default;
    };

    template <typename rnd_state>
    class NMCH_FE_K2_MM : public NMCH_FE_K1_MM<rnd_state> {
        public:
            NMCH_FE_K2_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual ~NMCH_FE_K2_MM() = default;
    };

    class NMCH_FE_K2_PHILOX_MM : public NMCH_FE_K1_MM<curandStatePhilox4_32_10_t> {
    public:
        NMCH_FE_K2_PHILOX_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
        virtual void compute() override;
        virtual ~NMCH_FE_K2_PHILOX_MM() = default;
    };

    template <typename rnd_state>
    class NMCH_FE_K1_PgM : public NMCH_FE_K1<rnd_state> {
        public:
            NMCH_FE_K1_PgM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual void init(unsigned long long seed) override;
            virtual ~NMCH_FE_K1_PgM() = default;
    };
    
    template <typename rnd_state>
    class NMCH_FE_K1_PiM : public NMCH_FE_K1<rnd_state> {
        public:
            NMCH_FE_K1_PiM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual void init(unsigned long long seed) override;
            virtual void finalize() override;
            virtual ~NMCH_FE_K1_PiM() = default;
        private:
            float *result;
    };

} // nmch::methods

#endif // "NMCH_FW_EULER_HPP"
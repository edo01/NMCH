#ifndef NMCH_EM_EULER_HPP
#define NMCH_EM_EULER_HPP

#include "NMCH/methods/NMCH.hpp"
#include <curand_kernel.h>


namespace nmch::methods
{
    /**
     * Abstract class for the Forward Euler method
     *
     * The nomeclature is the following:
     * - NMCH: Namespace for the Monte Carlo methods
     * - EM: Forward Euler
     * - KX: Version of the kernel
     * - YY: Type of Memory Management
     */
    template <typename rnd_state>
    class NMCH_EM_K1 : public NMCH<rnd_state>
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
            NMCH_EM_K1(int NTPB, int NB, float T, float S_0, float v_0, float r, float k,float rho, float theta, float sigma, int N);
            virtual void finalize() override;
            virtual void print_stats() override;
            virtual ~NMCH_EM_K1() = default;

            /**
             * @return execution time
             */
            float get_execution_time() const { return Tim_exec; }

            float get_err() const
            {
                float err = 1.96 * sqrt((double)(1.0f / (this->state_numbers - 1)) * (this->state_numbers*this->variance - 
                            (this->strike_price * this->strike_price)))/sqrt((double)this->state_numbers);
                return err;
            }
            
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
    class NMCH_EM_K1_MM : public NMCH_EM_K1<rnd_state> {
        public:
            NMCH_EM_K1_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual void init(unsigned long long seed) override;
            virtual ~NMCH_EM_K1_MM() = default;
    };

    template <typename rnd_state>
    class NMCH_EM_K2_MM : public NMCH_EM_K1_MM<rnd_state> {
        public:
            NMCH_EM_K2_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual ~NMCH_EM_K2_MM() = default;
    };

    template <typename rnd_state>
    class NMCH_EM_K3_MM : public NMCH_EM_K2_MM<rnd_state> {
        public:
            NMCH_EM_K3_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual ~NMCH_EM_K3_MM() = default;
    };
    
} // nmch::methods

#endif // "NMCH_FW_EULER_HPP"
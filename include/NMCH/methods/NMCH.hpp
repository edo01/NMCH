#ifndef NMCH_HPP
#define NMCH_HPP


#include <stdio.h>
#include "NMCH/random/random.hpp"
#include "NMCH/utils/utils.hpp"

namespace nmch::methods
{
    /**
     * NMCH is the base abstract class for giving the interface of different optimization strategies.
     * All the NMCH methods should have the same structure: the user creates a new NMCH object,
     * initializes it, computes the price of the option and the volatility at time T, and finalizes it.
     *
     * If the one wants to create a new implmentation, he should inherit from this class and implement the
     * necessary methods. 
     *
     * If the user wants to use the same method with different parameters, he can use the set methods and
     * call multiple times the compute method avoiding the initialization and finalization of the memory,
     * since the random states are kept in the memory.
     * 
     * Using this interface, it is easier to compare different methods and to create new ones, optimizing 
     * different aspects of the computation: memory management, shared memory, etc.
     *
     * Furthermore, the user can easily change the random number generator using templates.
     */
    template <typename rnd_state>
    class NMCH {
        public:
            /**
            * @param NTPB Number of threads per block
            * @param NB Number of blocks
            * @param T Time of the maturity
            * @param S_0 Spot values
            * @param r risk-free interest rate  
            * @param k mean reversion rate of the volatility
            * @param theta long-term variance
            * @param sigma volatility of the variance
            * @param N Number of time steps            
            */
            NMCH(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);

            /**
             * Compute the price of the option and the variance at time T
             */
            virtual void compute() = 0;
            /**
             * Print the statistics of the computation, such as the time of the computation, 
             * the number of threads, the number of blocks, etc.
             */
            virtual void print_stats();
            /**
             * Initializate the computation, such as the random states of the threads and the memory allocation.
             */
            virtual void init(unsigned long long seed) = 0;
            /**
             * Finalize the computation, such as the memory deallocation.
             */
            virtual void finalize() = 0;
            
            /**
            * @return the price of the option
            */
            float get_strike_price() const { return strike_price; }

            /**
            * @return the variance at time T
            */
            float get_variance() const { return variance; }

            /**
             * Set methods for the exploration of the parameters
             */

            void set_k(float k) { this->k = k; }

            void set_theta(float theta) { this->theta = theta; }    

            void set_sigma(float sigma) { this->sigma = sigma; }

            virtual ~NMCH() = default;

        protected:
            /* Number of threads per block */
            int NTPB;
            /* Number of blocks */
            int NB;
            /* Time of the maturity */
            float T;
            /* Initial price */
            float S_0;
            /* Initial variance */
            float v_0;
            /* Market price at time T - since it is at-the-money S_0=K */
            float K;
            /* risk-free interest rate */
            float r;
            /* mean reversion rate of the variance */
            float k;
            /* correlation */
            float rho;
            /* long-term variance */
            float theta;
            /* volatility of the variance */
            float sigma;
            /* Number of time steps */
            int N;
            /* time step */
            float dt;
            /* final strike_price */
            float strike_price;
            /* final variance*/
            float variance;
    };

} // namespace nmch::methods


#endif // NMCH_HPP
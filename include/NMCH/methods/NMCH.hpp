#ifndef NMCH_CUH
#define NMCH_CUH

/**
 * ## Monte Carlo Simulation of Heston Model
 * 
 * The Heston model for asset pricing has been widely examined in the literature. Under this model, the dynamics of the asset 
 * price $S_t$ and the variance $v_t$ are governed by the following system of stochastic differential equations:
 * 
 * $$
 * dS_t = r S_t \, dt + \sqrt{v_t} S_t \, dZ_t
 * $$
 * $$
 * dv_t = \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t
 * $$
 * $$
 * Z_t = \rho W_t + \sqrt{1 - \rho^2} Z_t
 * $$
 * 
 * Where:
 * - The spot values $S_0 = 1$ and $v_0 = 0.1$,
 * - $r$ is the risk-free interest rate, assumed to be $r = 0$,
 * - $\kappa$ is the mean reversion rate of the variance,
 * - $\theta$ is the long-term variance,
 * - $\sigma$ is the volatility of variance,
 * - $W_t$ and $Z_t$ are independent Brownian motions.
 * 
 * In this project, we aim to compare two distinct methods for simulating an at-the-money call option (where "at-the-money" here means $K = S_0 = 1$) at maturity $T = 1$ under the Heston model. The option has a payoff given by $f(x) = (x - K)^+$, so we want to simulate with Monte Carlo the expectation $E[f(S_T)] = E[(S_1 - 1)^+]$. 
 * 
 * This comparison will focus on the efficiency and accuracy of each simulation method in pricing the call option within the stochastic volatility framework of the Heston model.
 * 
 * We begin with the Euler discretization scheme, which updates the asset price $S_t$ and the volatility $v_t$ at each time step as follows:
 * 
 * $$
 * S_{t+\Delta t} = S_t + r S_t \Delta t + \sqrt{v_t} S_t \sqrt{\Delta t} \left( \rho G_1 + \sqrt{1 - \rho^2} G_2 \right)
 * $$
 * 
 * $$
 * v_{t+\Delta t} = g \left( v_t + \kappa (\theta - v_t) \Delta t + \sigma \sqrt{v_t} \sqrt{\Delta t} G_1 \right)
 * $$
 * 
 * where $G_1$ and $G_2$ are independent standard normal random variables, and the function $g$ is either taken to be equal to $(\cdot)^+$ or to $|\cdot|$.
 * 
 */



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
             * @return the bias
             */
            float get_bias() const 
            { 
                float real_price = this->S_0 * nmch::utils::NP((this->r + 0.5 * this->sigma * this->sigma)/this->sigma) -
                                        this->K * expf(-this->r) * nmch::utils::NP((this->r - 0.5 * this->sigma * this->sigma) /
                                        this->sigma);
                return abs((this->strike_price - real_price)/real_price);
            }

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


#endif // NMCH_CUH
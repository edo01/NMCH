#ifndef NMCH_FW_EULER_HPP
#define NMCH_FW_EULER_HPP

#include <curand_kernel.h>
#include "NMCH/methods/NMCH.hpp"
#include "NMCH/utils/utils.hpp"


namespace nmch::methods
{   
    /**
     * Abstract class for the Forward Euler method
     *
     * The nomeclature is the following:
     * - NMCH: Namespace for the Monte Carlo methods
     * - FE: Forward Euler
     * - KX: Version of the kernel
     * - YY: Type of Memory Management
     */
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

            /**
             * @return execution time
             */
            float get_execution_time() const { return Tim_exec; }

            /**
             * @return the estimation of the error of the simulation
             *         using the confidence interval: 
                       \( 1.96 * sqrt((1/(N-1)) * (N*variance - strike_price^2))/sqrt(N) \)
             */
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

    /**
     * Memory managed version of the kernel.
     * We saw that the use of different memory spaces does not affect the performance
     * of the code so the exploration in the type of memory doesn't help. In fact,
     * the communication between CPU and GPU is not significant, since we are moving only two floats
     * and the random states of the threads remain on the device.
     *
     * So for simplicity we will then use Memory Management for the rest of the optimizations. 
     *
     * with N  = 10000 and NTPB = 1024, NB = 512
     *  -NMCH_FE_K1_PgM:  
	 *   Execution time 52.874241 ms
	 *   Initialization time 6.773760 ms
     * - NMCH_FE_K1_PiM
     * 	 Execution time 52.875263 ms
     * 	 Initialization time 7.162592 ms
     * - NMCH_FE_K1_MM
     * 	 Execution time 52.882721 ms
     * 	 Initialization time 7.224960 ms
     *
     * K1 version of the kernel has the following features:
     * - It uses the Forward Euler method
     * - It reduces the results of the simulation on the device
     * - It saves the state of the random number generator   
     * - It uses aggregated calls to the random number generator to reduce the overhead of the calls
     */
    template <typename rnd_state>
    class NMCH_FE_K1_MM : public NMCH_FE_K1<rnd_state> {
        public:
            NMCH_FE_K1_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual void init(unsigned long long seed) override;
            virtual ~NMCH_FE_K1_MM() = default;
    };

    /**
     * In this version version we optimize the reduction by using warp shuffle instructions. 
     * Using warp shuffle instructions has the following advantages:
     * - reduces the shared memory pressure: the shared memory is limited only to the size of the warp
     * - Whereas to communicate a data item from a register in thread B to a register in thread A via shared memory
     *  requires at least 2 steps (a shared load instruction and a shared store instruction, and probably also a 
     *  synchronization step), the same communication via warp shuffle requires a single operation/instruction.
     *
     * Testing just the reduction using 102.400.000 threads takes 4.533248 ms when using the classic one,
     * while using warp reduction takes have 2.750464 ms. When using 1.024.000.000 threads we have 42.272766 ms
     * and 24.312481 ms respectively. However the reduction takes a small part of the total time, so the overall
     * performance is not deeply affected. But with warp reduction we can reduce the shared memory pressure and 
     * allow to further optimize the kernel. 
     *
     * K2 version of the kernel has the same features of the K1 version but it uses warp shuffle instructions for 
     * the reduction.
     */
    template <typename rnd_state>
    class NMCH_FE_K2_MM : public NMCH_FE_K1_MM<rnd_state> {
        public:
            NMCH_FE_K2_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual ~NMCH_FE_K2_MM() = default;
    };

    /**
     * This version is specialized for the Philox_4x32_10 number generator. This random generator allows to generate 4 
     * random numbers at the same time.
     *
     * This version allows to have 72.066048 ms against the normal version always using philox4_32_10 which has 85.052193 ms
     * Nevertheless, this number generator is slower than xorwow, which takes 53.237823 ms.
     */
    class NMCH_FE_K2_PHILOX_MM : public NMCH_FE_K1_MM<curandStatePhilox4_32_10_t> {
    public:
        NMCH_FE_K2_PHILOX_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
        virtual void compute() override;
        virtual ~NMCH_FE_K2_PHILOX_MM() = default;
    };

    /**
     * This version exploits the reduction of the shared memory pressure achieved by using warp shuffle instructions.
     * We can in fact use the shared memory to store the states of the random number generator instead of the private memory.
     * Actually, private memory can be stored also in global memory when the register pressure is too high. Saving the states 
     * in the shared memory allows to reduce the overhead of the calls to the random number generator when the registers are not 
     * enough.
     */
    template <typename rnd_state>
    class NMCH_FE_K3_MM : public NMCH_FE_K2_MM<rnd_state> {
        public:
            NMCH_FE_K3_MM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual ~NMCH_FE_K3_MM() = default;
    };

    /**
     * Paged Memory version of the kernel
     */
    template <typename rnd_state>
    class NMCH_FE_K1_PgM : public NMCH_FE_K1<rnd_state> {
        public:
            NMCH_FE_K1_PgM(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N);
            virtual void compute() override;
            virtual void init(unsigned long long seed) override;
            virtual ~NMCH_FE_K1_PgM() = default;
    };
    
    /**
     * Pinned Memory version of the kernel
     */
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
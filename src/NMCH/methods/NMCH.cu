#include "NMCH/methods/NMCH.hpp"

namespace nmch::methods {

    template <typename rnd_state>
    NMCH<rnd_state>::NMCH(int NTPB, int NB, float T, float S_0, float K, float sigma, float r, int N)
    : NTPB(NTPB), NB(NB), T(T), S_0(S_0), K(K), sigma(sigma), r(r), N(N)
    {
        dt = sqrt(T/N);
        state_numbers = NTPB * NB;
    };

    
    template <typename rnd_state>
    void NMCH<rnd_state>::init_curand_state()
    {
	    nmch::random::init_curand_state_k<<<NB, NTPB>>>(states);
    };

    template <typename rnd_state>
    void NMCH<rnd_state>::allocate_memory()
    {
        cudaMalloc(&sum, 2 * sizeof(float));
        cudaMemset(sum, 0, 2 * sizeof(float));
        cudaMalloc(&states, state_numbers * sizeof(rnd_state)); // is the total number of state
    };

    template <typename rnd_state>
    void NMCH<rnd_state>::free_memory()
    {
        cudaFree(sum);
        cudaFree(states);
    };

    template <typename rnd_state>
    void NMCH<rnd_state>::print_stats()
    {
        int n = NTPB * NB;
        // for now
        printf("The estimated price is equal to %f\n", sum[0]);
        printf("error associated to a confidence interval of 95%% = %f\n",
            1.96 * sqrt((double)(1.0f / (n - 1)) * (n*sum[1] - (sum[0] * sum[0])))/sqrt((double)n));
        printf("The true price %f\n", S_0 * nmch::utils::NP((r + 0.5 * sigma * sigma)/sigma) -
                                        K * expf(-r) * nmch::utils::NP((r - 0.5 * sigma * sigma) / sigma));
        //printf("Execution time %f ms\n", Tim);
    }; 

    template class NMCH<curandStateXORWOW_t>;
    template class NMCH<curandStateMRG32k3a_t>;
    template class NMCH<curandStatePhilox4_32_10_t>;
}
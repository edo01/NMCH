#include "NMCH/methods/NMCH.hpp"

namespace nmch::methods {

    template <typename rnd_state>
    NMCH<rnd_state>::NMCH(int NTPB, int NB, float T, float S_0, float v_0, float r, float k, float rho, float theta, float sigma, int N)
    : NTPB(NTPB), NB(NB), T(T), S_0(S_0), K(S_0), v_0(v_0), r(r), k(k), rho(rho), theta(theta), sigma(sigma), N(N)
    {
        dt  = T/N;
    };

    template <typename rnd_state>
    void NMCH<rnd_state>::print_stats()
    {
        // In this base version we just print the paramaters and the result
        printf("Base parameters:\n");
        printf("NTBP    = %d\n",   NTPB);
        printf("NB      = %d\n",     NB);
        printf("T       = %f\n",     T);
        printf("S_0,K   = %f\n",     S_0);
        printf("v_0     = %f\n",     v_0);
        printf("r       = %f\n",     r);
        printf("k       = %f\n",     k);
        printf("theta   = %f\n",     theta);
        printf("sigma   = %f\n",     sigma);
        printf("N       = %d\n",     N);
        printf("dt      = %f\n",     dt);
    }; 

    template class NMCH<curandStateXORWOW_t>;
    template class NMCH<curandStateMRG32k3a_t>;
    template class NMCH<curandStatePhilox4_32_10_t>;
}
#include "NMCH/methods/NMCH_fw_euler.hpp"
#include <curand_kernel.h>

using namespace nmch::methods;

int main(int argc, char **argv)
{
    int NTPB = 1024;
	int NB = 512;
	int n = NB * NTPB;
	float T = 1.0f;
	float S_0 = 50.0f;
	float K = S_0;
	float sigma = 0.2f;
	float r = 0.1f;
	int N = 100;
	float dt = sqrtf(T/N);
	float *sum;


    NMCH_fw_euler<curandStateXORWOW_t> nmch(NTPB, NB, T, S_0, K, sigma, r, N);
    nmch.compute();
    return 0;
}
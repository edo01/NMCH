#include "NMCH/methods/NMCH_fw_euler.hpp"
//#include <curand_kernel.h>

using namespace nmch::methods;

int main(int argc, char **argv)
{
    int NTPB = 1024;
	int NB = 512;
//	int n = NB * NTPB;
	float T = 1.0f;
	float S_0 = 50.0f;
	float v_0 = 0.1f;
	//float K = S_0;
	float r = 0.0f;
	float k = 0.5f;
	float theta = 0.1f;
	float sigma = 0.3f;
	int N = 1000;

	// (int NTPB, int NB, float T, float S_0, float r, float k, float theta, float sigma, int N)
    NMCH_fw_euler<curandStateXORWOW_t> nmch(NTPB, NB, T, S_0, v_0, r, k, theta, sigma, N);
    nmch.init();
	nmch.compute();
	nmch.print_stats();
	nmch.finalize();
    return 0;
}
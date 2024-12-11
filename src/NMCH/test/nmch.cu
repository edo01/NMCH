#include <string>
#include "NMCH/methods/NMCH_FE.hpp"
#include "NMCH/methods/NMCH_EM.hpp"

#include <curand_kernel.h>
#include <cuda_runtime.h>

/**
- default parameters
- N       = 10000

- NMCH_FE_K1_PgM:  
	Execution time 52.874241 ms
	Initialization time 6.773760 ms
- NMCH_FE_K1_PiM
	Execution time 52.875263 ms
	Initialization time 7.162592 ms
- NMCH_FE_K1_MM
	Execution time 52.882721 ms
	Initialization time 7.224960 ms

from this first analysis, it is clear that the use of different memory spaces does not affect the performance
of the code so we shouldn't push in this direction. For semplicity we will then use Memory Management for 
the rest of the project.

This is justified by the fact that the communication between CPU and GPU is not significant, since we are 
moving only two floats.
 */

/**
 * reduction data with 1024*100.000 = 102.400.000 threads
 * using normal reduction we have 4.533248 ms while using warp reduction 
 * we have 2.750464 ms. 
 * while using 1.024.000.000 threads we have 42.272766 ms and 24.312481 ms
 * respectively. 
 */

/**
 * using curand_normal4 in FE allows to have 72.066048 ms against the normal version always using 
 * philox4_32_10 which has 85.052193 ms and the normal 53.237823 ms using xorwow.
 */

/**
	presentation ideas: class hierarchy and speedup obtained with each strategy and why we chose a specific
	path.
 */
using namespace nmch::methods;

int main(int argc, char **argv)
{

	int NTPB = 512;
	int NB = 512;
	float T = 1.0f;
	float S_0 = 1.0f;
	float v_0 = 0.1f;
	float r = 0.0f;
	float k = 0.5f;
	float rho = -0.7;
	float theta = 0.1f;
	float sigma = 0.3f;
	int N = 1000;
	unsigned long long seed = 1234;
	std::string method = "fe"; // default method

	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--NTPB") == 0 && i + 1 < argc) {
			NTPB = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--NB") == 0 && i + 1 < argc) {
			NB = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) {
			T = atof(argv[++i]);
		} else if (strcmp(argv[i], "--S_0") == 0 && i + 1 < argc) {
			S_0 = atof(argv[++i]);
		} else if (strcmp(argv[i], "--v_0") == 0 && i + 1 < argc) {
			v_0 = atof(argv[++i]);
		} else if (strcmp(argv[i], "--r") == 0 && i + 1 < argc) {
			r = atof(argv[++i]);
		} else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
			k = atof(argv[++i]);
		} else if (strcmp(argv[i], "--rho") == 0 && i + 1 < argc) {
			rho = atof(argv[++i]);
		} else if (strcmp(argv[i], "--theta") == 0 && i + 1 < argc) {
			theta = atof(argv[++i]);
		} else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
			sigma = atof(argv[++i]);
		} else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
			N = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
			seed = strtoull(argv[++i], nullptr, 10);
		} else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
			method = argv[++i];
		} else if (strcmp(argv[i], "--help") == 0) {
			printf("Usage: %s [options]\n", argv[0]);
			printf("Options:\n");
			printf("  --NTPB <int>       Number of threads per block (default: 1024)\n");
			printf("  --NB <int>         Number of blocks (default: 512)\n");
			printf("  --T <float>        Time period (default: 1.0)\n");
			printf("  --S_0 <float>      Initial stock price (default: 1.0)\n");
			printf("  --v_0 <float>      Initial volatility (default: 0.1)\n");
			printf("  --r <float>        Risk-free rate (default: 0.0)\n");
			printf("  --k <float>        Mean reversion rate (default: 0.5)\n");
			printf("  --rho <float>      Correlation (default: -0.7)\n");
			printf("  --theta <float>    Long-term volatility (default: 0.1)\n");
			printf("  --sigma <float>    Volatility of volatility (default: 0.3)\n");
			printf("  --N <int>          Number of time steps (default: 50)\n");
			printf("  --seed <ull>       Random seed (default: 1234)\n");
			printf("  --method <string>  Method to use: fe or em (default: fe)\n");
			printf("  --help             Display this help message\n");
			return 0;
		}
	}

	if (method == "fe") {
		//NMCH_FE_K1_MM<curandStateXORWOW_t> nmch1(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);
		//NMCH_FE_K2_MM<curandStateXORWOW_t> nmch(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);
		//NMCH_FE_K2_PHILOX_MM nmch2P(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);
		NMCH_FE_K3_MM<curandStatePhilox4_32_10_t> nmch(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);

		nmch.init(seed);
		nmch.compute();
		nmch.print_stats();
		nmch.finalize();
		
	} else if (method == "em") {
		//NMCH_EM_K1_MM<curandStateXORWOW_t> nmch(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);
		//NMCH_EM_K2_MM<curandStateXORWOW_t> nmch(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);

		NMCH_EM_K3_MM<curandStatePhilox4_32_10_t> nmch(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);
		nmch.init(seed);
		nmch.compute();
		nmch.print_stats();
		nmch.finalize();
	} else {
		printf("Unknown method: %s\n", method.c_str());
		return 1;
	}
    return 0;
}
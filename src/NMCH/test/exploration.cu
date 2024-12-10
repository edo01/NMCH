#include <string>
#include "NMCH/methods/NMCH_FE.hpp"
#include "NMCH/methods/NMCH_EM.hpp"

#include <curand_kernel.h>
#include <cuda_runtime.h>

using namespace nmch::methods;

int main(int argc, char **argv)
{

	int NTPB = 512; // using shared memory
	// number of simulation paths to get from the command line
	int NB = 20;
	float T = 1.0f;
	float S_0 = 1.0f;
	float v_0 = 0.1f;
	float r = 0.0f;
	float rho = -0.7;
	int N = 100;
	unsigned long long seed = 1234;
	
	// default parameters
	float k = 0.5f;
	float theta = 0.1f;
	float sigma = 0.3f;

    // exploration parameters
    // k     ->   [0.1 10]
    // theta -> [0.01 0.5]
    // sigma    [0.1  1]
	// 20 k theta<sigma^2

	// set the bounds of the exploration
	float k_min = 0.1f, k_max = 10.0f;
	float theta_min = 0.01f, theta_max = 0.5f;
	float sigma_min = 0.1f, sigma_max = 1.0f;

	float sigma_step = (sigma_max - sigma_min)/5;
	float theta_step = (theta_max - theta_min)/5;
	float k_step = (k_max - k_min)/5;

	NMCH_FE_K3_MM<curandStateXORWOW_t> nmch_fe(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);
	NMCH_EM_K3_MM<curandStateXORWOW_t> nmch_em(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);

	nmch_fe.init(seed);
	nmch_em.init(seed);

	printf("method, k, theta, sigma, execution_time, bias\n");
	
	for(sigma = sigma_min; sigma <= sigma_max; sigma += sigma_step) {
		for(theta = theta_min; theta <= theta_max; theta += theta_step) {
			for(k = k_min; k <= k_max; k += k_step) {
				
				// the variance of the FE is too small otherwise
				if(20*k*theta < sigma*sigma ) continue;

				nmch_fe.set_theta(theta);
				nmch_fe.set_sigma(sigma);
				nmch_fe.set_k(k);
				nmch_fe.compute();

				float execution_time = nmch_fe.get_execution_time();
				float bias = nmch_fe.get_bias();
				printf("fe, %f, %f, %f, %f, %f\n", k, theta, sigma, execution_time, bias);
			}
		}
	}
		
	for(sigma = sigma_min; sigma <= sigma_max; sigma += sigma_step) {
		for(theta = theta_min; theta <= theta_max; theta += theta_step) {
			for(k = k_min; k <= k_max; k += k_step) {

				// to guarantee a good comparison
				if(20*k*theta < sigma*sigma ) continue;

				nmch_em.set_theta(theta);
				nmch_em.set_sigma(sigma);
				nmch_em.set_k(k);
				nmch_em.compute();

				float execution_time = nmch_em.get_execution_time();
				float bias = nmch_em.get_bias();
				printf("em, %f, %f, %f, %f, %f\n", k, theta, sigma, execution_time, bias);
			}
		}
	}

	nmch_em.finalize();
	nmch_fe.finalize();

    return 0;
}
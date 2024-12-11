#include <string>
#include "NMCH/methods/NMCH_FE.hpp"
#include "NMCH/methods/NMCH_EM.hpp"

#include <curand_kernel.h>
#include <cuda_runtime.h>

using namespace nmch::methods;

/**
 * Here we explore the parameter space of the NMCH method.
 * Different values of k, theta and sigma are tested to see how they affect the execution time
 * and the error of the method.
 *
 * The state of the random number generator is saved after each run so that we avoid to initialize 
 * it and to save time.
 *
 * The results are printed in a csv format so that they can be easily imported in python for further 
 * analysis. 
 */
int main(int argc, char **argv)
{
	// using shared memory so we need to set the number of threads per block <= 512
	int NTPB = 512;
	int NB = 40;
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
	float theta_step = (theta_max - theta_min)/20;
	float k_step = (k_max - k_min)/20;

	NMCH_FE_K3_MM<curandStateXORWOW_t> nmch_fe(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);
	NMCH_EM_K3_MM<curandStateXORWOW_t> nmch_em(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);

	nmch_fe.init(seed);
	nmch_em.init(seed);

	/** #############################################################
		 					FE EXPLORATION
		############################################################# 
	 */

	// the first run is always slow so we compute it here 
	// an avoid to pollute the exploration
	nmch_fe.compute();

	printf("method, k, theta, sigma, execution_time, err\n");
	
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
				float err = nmch_fe.get_err();
				printf("fe, %f, %f, %f, %f, %f\n", k, theta, sigma, execution_time, err);
			}
		}
	}

	/** #############################################################
		 				Exact Method EXPLORATION
		############################################################# 
	 */


	// the first run is always slow so we compute it here 
	// an avoid to pollute the exploration
	nmch_em.compute();
		
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
				float err = nmch_em.get_err();
				printf("em, %f, %f, %f, %f, %f\n", k, theta, sigma, execution_time, err);
			}
		}
	}

	nmch_em.finalize();
	nmch_fe.finalize();

    return 0;
}
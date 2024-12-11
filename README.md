# NMCH - Numerical Monte Carlo Simulation of Heston Model
A CUDA implementation of the Heston Model using Monte Carlo Simulation.

## Introduction
The Heston model for asset pricing has been widely examined in the literature. Under this model, the dynamics of the asset 
price $S_t$ and the variance $v_t$ are governed by the following system of stochastic differential equations:

$$
dS_t = r S_t \, dt + \sqrt{v_t} S_t \, dZ_t
$$
$$
dv_t = \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t
$$
$$
Z_t = \rho W_t + \sqrt{1 - \rho^2} Z_t
$$

Where:
- The spot values $S_0 = 1$ and $v_0 = 0.1$,
- $r$ is the risk-free interest rate, assumed to be $r = 0$,
- $\kappa$ is the mean reversion rate of the volatility,
- $\theta$ is the long-term volatility,
- $\sigma$ is the volatility of volatility,
- $W_t$ and $Z_t$ are independent Brownian motions.

In this project, we aim to compare two distinct methods for simulating an at-the-money call option (where "at-the-money" here means $K = S_0 = 1$) at maturity $T = 1$ under the Heston model. The option has a payoff given by $f(x) = (x - K)^+$, so we want to simulate with Monte Carlo the expectation $E[f(S_T)] = E[(S_1 - 1)^+]$. 

This comparison will focus on the efficiency and accuracy of each simulation method in pricing the call option within the stochastic volatility framework of the Heston model.

We begin with the Euler discretization scheme, which updates the asset price $S_t$ and the volatility $v_t$ at each time step as follows:

$$
S_{t+\Delta t} = S_t + r S_t \Delta t + \sqrt{v_t} S_t \sqrt{\Delta t} \left( \rho G_1 + \sqrt{1 - \rho^2} G_2 \right)
$$

$$
v_{t+\Delta t} = g \left( v_t + \kappa (\theta - v_t) \Delta t + \sigma \sqrt{v_t} \sqrt{\Delta t} G_1 \right)
$$

where $G_1$ and $G_2$ are independent standard normal random variables, and the function $g$ is either taken to be equal to $(\cdot)^+$ or to $|\cdot|$.

## Compiling and Running
To build the executable run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
```
two executables will be generated: `NMCH` and `exploration`. The first one is the main program that will run the Monte Carlo simulation for the Heston model, and the second one is a program that will run a series of simulation for exploring the parameter space of the problem.

The `NMCH` program can be run with the following command from the `build` directory:
```bash
$ ./bin/NMCH --method em --N 1000 --NTPB 512 --NB 512
```
this command will run an exact simulation(em) with 512 threads per block and 512 blocks, discretizing the time in 1000 steps. The exact method used is based on the one proposed in "_Mark Broadie and Özgür Kaya. Exact simulation of stochastic volatilityand other affine jump diffusion processes. Operations research, 54(2):217–231, 2006._".

## Add your test
To use the methods implemented in this project, you can add a new file in the `src/NMCH/test` directory and modify the CMakeLists.txt file accordingly. Here a simple usage example:

```c++
#include "NMCH/methods/NMCH_FE.hpp"
#include "NMCH/methods/NMCH_EM.hpp"

#include <curand_kernel.h>
#include <cuda_runtime.h>

int main()
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

    // step 1 declare the method
    NMCH_FE_K3_MM<curandStateXORWOW_t> nmch(NTPB, NB, T, S_0, v_0, r, k, rho, theta, sigma, N);

    // step 2 initialize the simulation
    nmch.init(seed);
    // step 3 run the simulation
    nmch.compute();
    // step 4 print the results
    nmch.print_stats();
    // step 5 finalize the simulation
    nmch.finalize();
}
```
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

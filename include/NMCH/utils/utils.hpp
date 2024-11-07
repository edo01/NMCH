#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace nmch::utils{
    
    double NP(double x);

} // namespace nmch::utils

namespace nmch::utils::cuda{

    void checkCUDA(cudaError_t error, const char* file, int line);

    // Has to be defined in the compilation in order to get the correct value of the 
    // macros __FILE__ and __LINE__
    #define testCUDA(error) (checkCUDA(error, __FILE__ , __LINE__))

} // namespace nmch::utils::cuda

#endif // UTILS_HPP
#ifdef CU_UTILS_HPP
#define CU_UTILS_HPP

namespace nmch::utils
{
    void testCUDA(cudaError_t error, const char* file, int line);

    // Has to be defined in the compilation in order to get the correct value of the 
    // macros __FILE__ and __LINE__
    #define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

} // namespace nmch::utils

#endif // CU_UTILS_HPP

#include "NMCH/utils/cu_utils.hpp"

namespace nmch::utils {

    void testCUDA(cudaError_t error, const char* file, int line) {
        if (error != cudaSuccess) {
            printf("There is an error in file %s at line %d\n", file, line);
            exit(EXIT_FAILURE);
        }
    }

} // namespace nmch::utils
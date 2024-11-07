#include "NMCH/utils/utils.hpp"

namespace nmch::utils{

    double NP(double x){
        const double p = 0.2316419;
        const double b1 = 0.319381530;
        const double b2 = -0.356563782;
        const double b3 = 1.781477937;
        const double b4 = -1.821255978;
        const double b5 = 1.330274429;
        const double one_over_twopi = 0.39894228;
        double t;

        if (x >= 0.0) {
            t = 1.0 / (1.0 + p * x);
            return (1.0 - one_over_twopi * exp(-x * x / 2.0) * t * (t * (t *
                (t * (t * b5 + b4) + b3) + b2) + b1));
        }
        else {/* x < 0 */
            t = 1.0 / (1.0 - p * x);
            return (one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t *
                (t * b5 + b4) + b3) + b2) + b1));
        }
    }
}
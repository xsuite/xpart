#ifndef XPART_RANDOM_GENERATOR_H
#define XPART_RANDOM_GENERATOR_H
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
=========================================
=====  Rutherford Random Generator  =====
=========================================
*/

// TODO: how to optimise Newton's method??

// PDF of Rutherford distribution
/*gpufun*/
double ruth_PDF(double t, double A, double B){
    return (A/pow(t,2))*(exp(-B*t));
}

// CDF of Rutherford distribution
/*gpufun*/
double ruth_CDF(double t, double A, double B, double t0){
    return A*B*Exponential_Integral_Ei(-B*t0) + t0*ruth_PDF(t0, A, B)
         - A*B*Exponential_Integral_Ei(-B*t)  - t*ruth_PDF(t, A, B);
        
}

/*gpukern*/
void RandomGeneratorData_set_rutherford(RandomGeneratorData ran, double z, double emr, double upper_val){
    double c = 0.8561e3; // TODO: Where tha fuck does this come from??
    double A = pow(z,2);
    double B = c*pow(emr,2);
    double lower_val = RandomGeneratorData_get_rutherford_lower_val(ran);

    // Normalise PDF
    double N = ruth_CDF(upper_val, A, B, lower_val);
    RandomGeneratorData_set_rutherford_A(ran, A/N);
    RandomGeneratorData_set_rutherford_B(ran, B);
    RandomGeneratorData_set_rutherford_upper_val(ran, upper_val);
}

// Generate a random value weighted with a Rutherford distribution
/*gpufun*/
double RandomGenerator_get_double_ruth(RandomGeneratorData ran, LocalParticle* part){

    // get the parameters
    double x0     = RandomGeneratorData_get_rutherford_lower_val(ran);
    int8_t n_iter = RandomGeneratorData_get_rutherford_iterations(ran);
    double A      = RandomGeneratorData_get_rutherford_A(ran);
    double B      = RandomGeneratorData_get_rutherford_B(ran);
    
    if (A==0. || B==0.){
        // Not initialised
        return 0.;
    }

    // sample a random uniform
    double t = RandomGenerator_get_double(part);

    // initial estimate is the lower border
    double x = x0;

    // HACK to let iterations depend on sample to improve speed
    // based on Berylium being worst performing and hcut as in materials table
    // DOES NOT WORK
//     if (n_iter==0){
//         if (t<0.1) {
//             n_iter = 3;
//         } else if (t<0.35) {
//             n_iter = 4;
//         } else if (t<0.63) {
//             n_iter = 5;
//         } else if (t<0.8) {
//             n_iter = 6;
//         } else if (t<0.92) {
//             n_iter = 7;
//         } else if (t<0.96) {
//             n_iter = 8;
//         } else if (t<0.98) {
//             n_iter = 9;
//         } else {
//             n_iter = 10;
//         }
//     }

    // solve CDF(x) == t for x
    int8_t i = 1;
    while(i <= n_iter) {
        x = x - (ruth_CDF(x, A, B, x0)-t)/ruth_PDF(x, A, B);
        i++;
    }

    return x;
}


#endif

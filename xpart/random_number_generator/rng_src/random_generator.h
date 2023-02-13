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


#endif

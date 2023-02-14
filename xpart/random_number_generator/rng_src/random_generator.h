// copyright ############################### //
// This file is part of the Xpart Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XPART_RANDOM_GENERATOR_H
#define XPART_RANDOM_GENERATOR_H
#include <stdlib.h>
#include <math.h>
#include <time.h>


/*
=========================================
=======  Standard Distributions  ========
=========================================
*/

/*gpufun*/
double RandomGenerator_get_double(LocalParticle* part){
    uint32_t s1 = LocalParticle_get__rng_s1(part);
    uint32_t s2 = LocalParticle_get__rng_s2(part);
    uint32_t s3 = LocalParticle_get__rng_s3(part);
    uint32_t s4 = LocalParticle_get__rng_s4(part);

    double r = rng_get(&s1, &s2, &s3, &s4);

    LocalParticle_set__rng_s1(part, s1);
    LocalParticle_set__rng_s2(part, s2);
    LocalParticle_set__rng_s3(part, s3);
    LocalParticle_set__rng_s4(part, s4);

    return r;
}


/*gpufun*/
double RandomGenerator_get_double_exp(LocalParticle* part){
    double x1 = RandomGenerator_get_double(part);
    while(x1==0.0){
        x1 = RandomGenerator_get_double(part);
    }
    return -log(x1);
}


/*gpufun*/
double RandomGenerator_get_double_gauss(LocalParticle* part){
    double x1 = RandomGenerator_get_double(part);
    while(x1==0.0){
        x1 = RandomGenerator_get_double(part);
    }
    x1 = sqrt(-2.0*log(x1));
    double x2 = RandomGenerator_get_double(part);
    x2 = 2.0*PI*x2;
    double r = x1*sin(x2);
    return r;
}




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
void RandomGeneratorData_set_rutherford(RandomGeneratorData ran, double A, double B, double lower_val, double upper_val){
    // Normalise PDF
    double N = ruth_CDF(upper_val, A, B, lower_val);
    RandomGeneratorData_set_rutherford_A(ran, A/N);
    RandomGeneratorData_set_rutherford_B(ran, B);
    RandomGeneratorData_set_rutherford_lower_val(ran, lower_val);
    RandomGeneratorData_set_rutherford_upper_val(ran, upper_val);
}

// Generate a random value weighted with a Rutherford distribution
/*gpufun*/
double RandomGenerator_get_double_ruth(RandomGeneratorData rng, LocalParticle* part){

    // get the parameters
    double x0     = RandomGeneratorData_get_rutherford_lower_val(rng);
    int8_t n_iter = RandomGeneratorData_get_rutherford_iterations(rng);
    double A      = RandomGeneratorData_get_rutherford_A(rng);
    double B      = RandomGeneratorData_get_rutherford_B(rng);
    
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



/*
=========================================
======  Used to sample in python  =======
=========================================
*/

/*gpufun*/
void RandomGenerator_track_local_particle(RandomGeneratorData rng, LocalParticle* part0) {
    int64_t n_samples       = RandomGeneratorData_get__n_samples(rng);
    int8_t dist             = RandomGeneratorData_get__distribution(rng);
    double val;

    //start_per_particle_block (part0->part)
    int i;
    for (i=0; i<n_samples; ++i){
        if (dist==0) {
            val = RandomGenerator_get_double(part);
        } else if (dist==1) {
            val = RandomGenerator_get_double_exp(part);
        } else if (dist==2) {
            val = RandomGenerator_get_double_gauss(part);
        } else if (dist==3) {
            val = RandomGenerator_get_double_ruth(rng, part);
        } else {
            val = 0;
        }
        RandomGeneratorData_set__samples(rng, n_samples*LocalParticle_get_particle_id(part) + i, val);
    }
    //end_per_particle_block
}

#endif /* XPART_RANDOM_GENERATOR_H */

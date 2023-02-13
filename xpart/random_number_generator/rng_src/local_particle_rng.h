// copyright ############################### //
// This file is part of the Xpart Package.   //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef LOCALPARTICE_RNG_H
#define LOCALPARTICE_RNG_H

/*gpufun*/
double LocalParticle_generate_random_double(LocalParticle* part)
{
    uint32_t s1 = LocalParticle_get__rng_s1(part);
    uint32_t s2 = LocalParticle_get__rng_s2(part);
    uint32_t s3 = LocalParticle_get__rng_s3(part);

    double r = rng_get(&s1, &s2, &s3);

    LocalParticle_set__rng_s1(part, s1);
    LocalParticle_set__rng_s2(part, s2);
    LocalParticle_set__rng_s3(part, s3);

    return r;
}

/*gpufun*/
double LocalParticle_generate_random_double_exp(LocalParticle* part)
{
  return -log(LocalParticle_generate_random_double(part));
}

/*gpufun*/
double LocalParticle_generate_random_double_gauss(LocalParticle* part)
{
    double x1 = LocalParticle_generate_random_double(part);
    while(x1==0.0){
        x1 = LocalParticle_generate_random_double(part);
    }
    x1 = sqrt(-2.0*log(x1));
    double x2 = LocalParticle_generate_random_double(part);
    x2 = 2.0*3.1415926535897932384626433832795028841971693993751*x2;
    double r = x1*sin(x2);
    return r;
}

// Generate a random value weighted with a Rutherford distribution
/*gpufun*/
double RandomGenerator_get_double_rutherford(RandomGeneratorData ran, LocalParticle* part){

    // get the parameters
    double x0     = RandomGeneratorData_get_rutherford_lower_val(ran);
    int8_t n_iter = RandomGeneratorData_get_rutherford_iterations(ran);
    double A      = RandomGeneratorData_get_rutherford_A(ran);
    double B      = RandomGeneratorData_get_rutherford_B(ran);
    
    if (A==0 || B==0){
        // Not initialised
        return 0.
    }

    // sample a random uniform
    double t = LocalParticle_generate_random_double(part);

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

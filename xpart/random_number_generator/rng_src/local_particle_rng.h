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

#endif

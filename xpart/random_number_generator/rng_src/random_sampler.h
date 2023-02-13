// copyright ############################### //
// This file is part of the Xpart Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XPART_RANDOM_SAMPLER_H
#define XPART_RANDOM_SAMPLER_H
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*gpufun*/
void RandomSampler_track_local_particle(RandomSamplerData el, LocalParticle* part0) {
    int64_t n_samples       = RandomSamplerData_get__n_samples(el);
    int8_t dist             = RandomSamplerData_get_distribution(el);
    RandomGeneratorData rng = RandomSamplerData_getp_generator(el);
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
        RandomSamplerData_set__samples(el, n_samples*LocalParticle_get_particle_id(part) + i, val);
    }
    //end_per_particle_block
}

#endif /* XPART_RANDOM_SAMPLER_H */

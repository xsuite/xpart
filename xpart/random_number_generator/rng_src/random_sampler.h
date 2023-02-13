#ifndef XPART_RANDOM_SAMPLER_H
#define XPART_RANDOM_SAMPLER_H
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*gpufun*/
void RandomSampler_track_local_particle(RandomSamplerData el, LocalParticle* part0) {
    int64_t n_samples       = RandomSamplerData_get_n_samples(el);
    int8_t dist             = RandomSamplerData_get_distribution(el);
    RandomGeneratorData rng = RandomSamplerData_getp_generator(el);
    double val;

    //start_per_particle_block (part0->part)
    int i;
    for (i=0; i<n_samples; ++i){
        if (dist==0) {
            val = LocalParticle_generate_random_double(part);
        } else if (dist==1) {
            val = LocalParticle_generate_random_double_exp(part);
        } else if (dist==2) {
            val = LocalParticle_generate_random_double_gauss(part);
        } else if (dist==3) {
            val = RandomGenerator_get_double_rutherford(rng, part);
        } else {
            val = 0;
        }

        RandomSamplerData_set__samples(el, n_samples*LocalParticle_get_particle_id(part) + i, val);
    }
    //end_per_particle_block
}

#endif

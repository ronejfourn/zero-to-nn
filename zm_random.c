#include "zm_random.h"

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static pcg32_random_t RNG = { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL };

f32 zm_random() {
    return (f32)pcg32_random_r(&RNG) / (f32)UINT32_MAX;
}

f32 zm_random_n() {
    return zm_random() * 2 - 1;
}

f32 zm_random_r(f32 a, f32 b) {
    return zm_random() * (b - a) + a;
}

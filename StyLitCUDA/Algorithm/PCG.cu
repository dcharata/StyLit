#include "PCG.cuh"

namespace StyLitCUDA {
namespace PCG {

__device__ void advance(PCGState *rng) {
  rng->state = rng->state * 6364136223846793005ULL + rng->increment;
}

__device__ uint32_t output(uint64_t state) {
  return (uint32_t)(((state >> 22u) ^ state) >> ((state >> 61u) + 22u));
}

__device__ uint32_t rand(PCGState *rng) {
  uint64_t oldstate = rng->state;
  advance(rng);
  return output(oldstate);
}

__device__ void init(PCGState *rng, uint64_t seed, uint64_t stream) {
  rng->state = 0U;
  rng->increment = (stream << 1u) | 1u;
  advance(rng);
  rng->state += seed;
  advance(rng);
}

} /* namespace PCG */
} /* namespace StyLitCUDA */

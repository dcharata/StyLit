#ifndef PCG_H_
#define PCG_H_

#include "../Utilities/ImagePitch.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

namespace StyLitCUDA {

namespace PCG {

/**
 * @brief The PCGState struct This holds the state for PCG pseudorandom number generation. This is
 * copied from EBSynth.
 */
struct PCGState {
  uint64_t state;
  uint64_t increment;
};

__device__ void pcgAdvance(PCGState *rng);

__device__ uint32_t pcgOutput(uint64_t state);

__device__ uint32_t pcgRand(PCGState *rng);

__device__ void pcgInit(PCGState *rng, uint64_t seed, uint64_t stream);

// This is used to maintain a state for a pseudorandom number generator for each source pixel.
using RandomState = ImagePitch<PCGState>;

} /* namespace PCG */

} /* namespace StyLitCUDA */

#endif /* PCG_H_ */

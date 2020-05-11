#ifndef PCG_H_
#define PCG_H_

#include "../Utilities/Image.cuh"
#include "../Utilities/PyramidImage.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

namespace StyLitCUDA {

/**
 * @brief The PCGState struct This holds the state for PCG pseudorandom number generation. This is
 * copied from EBSynth.
 */
struct PCGState {
  uint64_t state;
  uint64_t increment;
};

namespace PCG {

__device__ void advance(PCGState *rng);

__device__ uint32_t output(uint64_t state);

__device__ uint32_t rand(PCGState *rng);

__device__ void init(PCGState *rng, uint64_t seed, uint64_t stream);

} /* namespace PCG */
} /* namespace StyLitCUDA */

#endif /* PCG_H_ */

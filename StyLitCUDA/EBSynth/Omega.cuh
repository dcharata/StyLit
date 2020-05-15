#ifndef OMEGA_H_
#define OMEGA_H_

#include "../Algorithm/NNFEntry.h"
#include "../Utilities/Coordinates.cuh"
#include "../Utilities/Image.cuh"
#include "../Utilities/PyramidImage.cuh"

namespace StyLitCUDA {
namespace Omega {

void initialize(const Image<NNFEntry> &nnf, Image<float> &omegas, const int patchSize);

__device__ void update(Image<float> &omegas, const int row, const int col, float difference,
                       const int patchSize);

__device__ float get(const Image<float> &omegas, const int row, const int col, const int patchSize);

} /* namespace Omega */
} /* namespace StyLitCUDA */

#endif /* OMEGA_H_ */

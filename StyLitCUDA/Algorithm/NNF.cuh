#ifndef NNF_H_
#define NNF_H_

#include "../Utilities/Coordinates.cuh"
#include "../Utilities/Image.cuh"
#include "../Utilities/PyramidImage.cuh"
#include "../Utilities/Vec.cuh"
#include "NNFEntry.h"
#include "PCG.cuh"

namespace StyLitCUDA {
namespace NNF {

template <typename T>
void randomize(Image<NNFEntry> &nnf, Image<PCGState> &random, const Image<T> &from,
               const Image<T> &to, const int patchSize, const Vec<float> &guideWeights,
               const Vec<float> &styleWeights);

template <typename T>
void recalculateErrors(Image<NNFEntry> &nnf, const Image<T> &from, const Image<T> &to,
                       const int patchSize, const Vec<float> &guideWeights,
                       const Vec<float> &styleWeights);

// Invalid rows/columns point to -1.
const int INVALID = -1;

/**
 * @brief invalidate Sets all the mappings in the NNF to be invalid.
 * @param nnf the NNF to invalidate
 */
void invalidate(Image<NNFEntry> &nnf);

/**
 * @brief upscale Upscales the given NNF by a factor of two.
 * @param from the NNF to read from (1x size)
 * @param to the NNF to write to (2x size)
 */
void upscale(const Image<NNFEntry> &from, Image<NNFEntry> &to, const int patchSize);

} /* namespace NNF */
} /* namespace StyLitCUDA */

#endif /* NNF_H_ */

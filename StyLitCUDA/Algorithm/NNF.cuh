#ifndef NNF_H_
#define NNF_H_

#include "../Utilities/Image.cuh"
#include "../Utilities/PyramidImage.cuh"
#include "NNFEntry.h"
#include "PCG.cuh"

namespace StyLitCUDA {
namespace NNF {

template <typename T>
void randomize(Image<NNFEntry> &nnf, Image<PCGState> &random, const Image<T> &from, const Image<T> &to, const int patchSize);

} /* namespace NNF */
} /* namespace StyLitCUDA */

#endif /* NNF_H_ */

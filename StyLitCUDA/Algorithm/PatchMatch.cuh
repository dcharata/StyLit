#ifndef PATCHMATCHER_H_
#define PATCHMATCHER_H_

#include "../Utilities/Image.cuh"
#include "NNFEntry.h"
#include "PCG.cuh"

namespace StyLitCUDA {
namespace PatchMatch {

template <typename T>
void run(Image<NNFEntry> &nnf, const Image<NNFEntry> *blacklist, const Image<T> &from,
         const Image<T> &to, const Image<PCGState> &random, const int patchSize,
         const int numIterations);

} /* namespace PatchMatch */
} /* namespace StyLitCUDA */

#endif /* PATCHMATCHER_H_ */

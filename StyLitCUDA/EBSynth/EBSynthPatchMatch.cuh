#ifndef EBSynthPatchMatch_H_
#define EBSynthPatchMatch__H_

#include "../Algorithm/NNFEntry.h"
#include "../Algorithm/PCG.cuh"
#include "../Utilities/Image.cuh"
#include "../Utilities/Vec.cuh"

namespace StyLitCUDA {
namespace EBSynthPatchMatch {

template <typename T>
void run(Image<NNFEntry> &nnf, Image<float> &omegas, const Image<T> &from, const Image<T> &to,
         const Image<PCGState> &random, const int patchSize, const int numIterations,
         const Vec<float> &weights);

} /* namespace EBSynthPatchMatch */
} /* namespace StyLitCUDA */

#endif /* EBSynthPatchMatch_H_ */

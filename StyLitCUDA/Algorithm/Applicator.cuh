#ifndef APPLICATOR_H_
#define APPLICATOR_H_

#include "../Utilities/Image.cuh"
#include "NNFEntry.h"

namespace StyLitCUDA {
namespace Applicator {

template <typename T>
void apply(const Image<NNFEntry> &nnf, Image<T> &from, const Image<T> &to, const int startChannel,
           const int endChannel, const int patchSize);

} /* namespace Applicator */
} /* namespace StyLitCUDA */

#endif /* APPLICATOR_H_ */

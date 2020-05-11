#ifndef NNF_H_
#define NNF_H_

#include "../Utilities/ImagePitch.cuh"

#include <limits>

namespace StyLitCUDA {

struct NNFEntry {
  int row = -1;
  int col = -1;
  float error = std::numeric_limits<float>::max();
};

using NNF = ImagePitch<NNFEntry>;

} /* namespace StyLitCUDA */

#endif /* NNF_H_ */

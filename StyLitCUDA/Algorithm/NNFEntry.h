#ifndef NNFENTRY_H_
#define NNFENTRY_H_

#include <limits>

namespace StyLitCUDA {

struct NNFEntry {
  int row = -1;
  int col = -1;
  float error = std::numeric_limits<float>::max();
};

} /* namespace StyLitCUDA */

#endif /* NNFENTRY_H_ */

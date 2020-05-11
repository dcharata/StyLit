#ifndef COORDINATOR_H_
#define COORDINATOR_H_

#include "../Interface/InterfaceInput.h"
#include "../Utilities/PyramidImage.cuh"
#include "NNFEntry.h"
#include "PCG.cuh"

#include <memory>

namespace StyLitCUDA {

template <typename T> class Coordinator {
public:
  Coordinator(InterfaceInput<T> &input);
  virtual ~Coordinator();

private:
  InterfaceInput<T> &input;

  // This contains the channels for A and A'.
  PyramidImage<T> a;

  // This contains the channels for B and B'.
  PyramidImage<T> b;

  // This image contains the state needed for per-pixel pseudorandom number generation.
  // It's big enough to work for both A and B.
  Image<PCGState> random;

  // This contains a forward NNF for each pyramid level.
  // (forward NNF: B-sized array of entries in A)
  PyramidImage<NNFEntry> forward;

  // This contains a reverse NNF for each pyramid level.
  // (reverse NNF: A-sized array of entries in B)
  PyramidImage<NNFEntry> reverse;
};

// This allows the C++ code to access the coordinator.
void runCoordinator_float(InterfaceInput<float> &input);

} /* namespace StyLitCUDA */

#endif /* COORDINATOR_H_ */

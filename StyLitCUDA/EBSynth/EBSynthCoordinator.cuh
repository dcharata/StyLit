#ifndef EBSYNTHCOORDINATOR_H_
#define EBSYNTHCOORDINATOR_H_

#include "../Algorithm/NNFEntry.h"
#include "../Algorithm/PCG.cuh"
#include "../Interface/InterfaceInput.h"
#include "../Utilities/PyramidImage.cuh"

#include <memory>

namespace StyLitCUDA {

template <typename T> class EBSynthCoordinator {
public:
  EBSynthCoordinator(InterfaceInput<T> &input);
  virtual ~EBSynthCoordinator();

private:
  InterfaceInput<T> &input;

  // This contains the channels for A and A'.
  PyramidImage<T> a;

  // This contains the channels for B and B'.
  PyramidImage<T> b;

  // This contains the omega values for each level.
  // It's the same size as A.
  PyramidImage<float> omegas;

  // This image contains the state needed for per-pixel pseudorandom number generation.
  // It's big enough to work for both A and B.
  Image<PCGState> random;

  // This contains a forward NNF for each pyramid level.
  // (forward NNF: B-sized array of entries in A)
  PyramidImage<NNFEntry> forward;
};

// This allows the C++ code to access the EBSynthCoordinator.
void runEBSynthCoordinator_float(InterfaceInput<float> &input);

} /* namespace StyLitCUDA */

#endif /* EBSYNTHCOORDINATOR_H_ */

#ifndef COORDINATOR_H_
#define COORDINATOR_H_

#include "../Interface/InterfaceInput.h"
#include "../Utilities/PyramidImage.cuh"

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
};

// This allows the C++ code to access the coordinator.
void runCoordinator_float(InterfaceInput<float> &input);

} /* namespace StyLitCUDA */

#endif /* COORDINATOR_H_ */

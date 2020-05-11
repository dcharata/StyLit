#ifndef INTERFACEINPUT_H_
#define INTERFACEINPUT_H_

#include "InterfaceImage.h"

#include <vector>

namespace StyLitCUDA {

template <typename T> struct InterfaceInput {
  // This is for the highest-resolution version of A.
  InterfaceImage<T> a;

  // This is for the highest-resolution version of A'.
  InterfaceImage<T> aPrime;

  // This is for the highest-resolution version of B.
  InterfaceImage<T> b;

  // This should have space allocated for the highest-resolution version of B'.
  InterfaceImage<T> bPrime;

  int numLevels;

  int patchSize;
};

} /* namespace StyLitCUDA */

#endif /* INTERFACEINPUT_H_ */

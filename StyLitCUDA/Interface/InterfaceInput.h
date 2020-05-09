#ifndef INTERFACEINPUT_H_
#define INTERFACEINPUT_H_

#include "InterfaceImage.h"

namespace StyLitCUDA {

template <typename T> struct InterfaceInput {
  InterfaceInput(const InterfaceImage<T> &a, const InterfaceImage<T> &aPrime,
                 const InterfaceImage<T> &b, InterfaceImage<T> &bPrime);
  virtual ~InterfaceInput() = default;

  // This is for the highest-resolution version of A.
  const InterfaceImage<T> &a;

  // This is for the highest-resolution version of A'.
  const InterfaceImage<T> &aPrime;

  // This is for the highest-resolution version of B.
  const InterfaceImage<T> &b;

  // This should have space allocated for the highest-resolution version of B'.
  InterfaceImage<T> &bPrime;
};

} /* namespace StyLitCUDA */

#endif /* INTERFACEINPUT_H_ */

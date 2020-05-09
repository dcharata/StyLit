#include "InterfaceInput.h"

namespace StyLitCUDA {

template <typename T>
InterfaceInput<T>::InterfaceInput(const InterfaceImage<T> &a, const InterfaceImage<T> &aPrime,
                                  const InterfaceImage<T> &b, InterfaceImage<T> &bPrime)
    : a(a), aPrime(aPrime), b(b), bPrime(bPrime) {}

template struct InterfaceInput<int>;
template struct InterfaceInput<float>;

} /* namespace StyLitCUDA */

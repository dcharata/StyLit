#include "InterfaceImage.h"

namespace StyLitCUDA {

template <typename T>
InterfaceImage<T>::InterfaceImage(const int rows, const int cols, const int numChannels,
                                  T *const data)
    : rows(rows), cols(cols), numChannels(numChannels), data(data) {}

template <typename T> T *InterfaceImage<T>::at(const int row, const int col) {
  const int index = row * cols + col;
  return &data[index * numChannels];
}

template <typename T> const T *InterfaceImage<T>::constAt(const int row, const int col) const {
  const int index = row * cols + col;
  return &data[index * numChannels];
}

template struct InterfaceImage<int>;
template struct InterfaceImage<float>;

} /* namespace StyLitCUDA */

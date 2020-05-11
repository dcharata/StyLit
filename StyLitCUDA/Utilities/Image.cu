#include "Image.cuh"

#include "../Interface/InterfaceImage.h"

namespace StyLitCUDA {

template <typename T>
Image<T>::Image(const int rows, const int cols, const int numChannels)
    : rows(rows), cols(cols), numChannels(numChannels) {}

template struct Image<int>;
template struct Image<float>;

} /* namespace StyLitCUDA */

#include "InterfaceImage.h"

template<typename T>
InterfaceImage<T>::InterfaceImage(const int rows, const int cols, const int channels,
                                  const T * const data)
    : rows(rows), cols(cols), channels(channels), data(data) {}

template struct InterfaceImage<int> ;
template struct InterfaceImage<float> ;

#include "PyramidImage.cuh"

namespace StyLitCUDA {

template <typename T>
PyramidImage<T>::PyramidImage(const int rows, const int cols, const int numChannels,
                                const int numLevels)
    : rows(rows), cols(cols), numChannels(numChannels), numLevels(numLevels) {}

template struct PyramidImage<int>;
template struct PyramidImage<float>;

} /* namespace StyLitCUDA */

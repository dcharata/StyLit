#include "PyramidImage.cuh"

#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {

template <typename T>
PyramidImage<T>::PyramidImage(const int rows, const int cols, const int numChannels,
                              const int numLevels)
    : rows(rows), cols(cols), numChannels(numChannels), numLevels(numLevels) {
  // Allocates on-device memory for each pyramid level.
  levels.clear();
  levels.emplace_back(rows, cols, numChannels);
  for (int level = 0; level < numLevels; level++) {
    if (level > 0) {
      const Image<T> &previous = levels[level - 1];
      levels.emplace_back(previous.rows / 2, previous.cols / 2, numChannels);
    }
    levels[level].allocate();
  }
}

template <typename T> PyramidImage<T>::~PyramidImage() {
  // Frees the on-device memory for each pyramid level.
  for (Image<T> &image : levels) {
    image.free();
  }
}

template struct PyramidImage<int>;
template struct PyramidImage<float>;

} /* namespace StyLitCUDA */

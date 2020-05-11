#include "PyramidImagePitch.cuh"

#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {

template <typename T>
PyramidImagePitch<T>::PyramidImagePitch(const int rows, const int cols, const int numChannels,
                                        const int numLevels)
    : rows(rows), cols(cols), numChannels(numChannels), numLevels(numLevels) {
  // Allocates on-device memory for each pyramid level.
  levels.clear();
  levels.emplace_back(rows, cols, numChannels);
  for (int level = 0; level < numLevels; level++) {
    if (level > 0) {
      const ImagePitch<T> &previous = levels[level - 1];
      levels.emplace_back(previous.rows / 2, previous.cols / 2, numChannels);
    }
    levels[level].allocate();
  }
}

template <typename T> PyramidImagePitch<T>::~PyramidImagePitch() {
  // Frees the on-device memory for each pyramid level.
  for (ImagePitch<T> &image : levels) {
    image.free();
  }
}

template struct PyramidImagePitch<int>;
template struct PyramidImagePitch<float>;

} /* namespace StyLitCUDA */

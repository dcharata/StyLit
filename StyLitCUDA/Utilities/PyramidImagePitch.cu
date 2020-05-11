#include "PyramidImagePitch.cuh"

#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {

template <typename T>
PyramidImagePitch<T>::PyramidImagePitch(const int rows, const int cols, const int numChannels,
                                        const int numLevels)
    : PyramidImage<T>(rows, cols, numChannels, numLevels), deviceLevels(nullptr) {}

template <typename T> void PyramidImagePitch<T>::allocate() {
  // Allocates device memory for deviceLevels.
  const size_t deviceLevelsSize = this->numLevels * sizeof(ImagePitch<T>);
  check(cudaMalloc(&deviceLevels, deviceLevelsSize));

  // Allocates temporary memory for hostLevels.
  ImagePitch<T> *hostLevels;
  check(cudaMallocHost(&hostLevels, deviceLevelsSize));

  // Populates hostLevels.
  // When implementing this, I ran into an issue where assigning directly to hostLevels[i] does not
  // copy what's needed to call functions on ImagePitch<T>. So in other words, calling
  // hostLevels[i]->allocate() would cause a segfault. This is a workaround, but clearly, I'm not
  // enough of a C++ programmer to really understand what's going on. Doing a memcpy into hostLevels
  // makes the function call work, but I assume then it won't work on the device, since the vtable
  // reference (or whatever it is) will no longer be valid.
  ImagePitch<T> currentLevel(this->rows, this->cols, this->numChannels);
  hostLevels[0] = currentLevel;
  for (int level = 0; level < this->numLevels; level++) {
    if (level > 0) {
      const ImagePitch<T> &lastLevel = hostLevels[level - 1];
      currentLevel = ImagePitch<T>(lastLevel.rows / 2, lastLevel.cols / 2, this->numChannels);
    }
    currentLevel.allocate();
    hostLevels[level] = currentLevel;
  }

  // Copies hostLevels to the device.
  check(cudaMemcpy((void *)deviceLevels, (void *)hostLevels, deviceLevelsSize,
                   cudaMemcpyHostToDevice));

  // Frees the temporary memory for hostLevels.
  check(cudaFreeHost(hostLevels));
}

template <typename T> void PyramidImagePitch<T>::free() {
  // Allocates and populates temporary memory for hostLevels.
  ImagePitch<T> *hostLevels;
  const size_t deviceLevelsSize = this->numLevels * sizeof(ImagePitch<T>);
  check(cudaMallocHost(&hostLevels, deviceLevelsSize));
  check(cudaMemcpy((void *)hostLevels, (void *)deviceLevels, deviceLevelsSize,
                   cudaMemcpyDeviceToHost));

  // Frees the images in deviceLevels, then deviceLevels itself.
  // The variable currentLevel is needed for essentially the same reason as in allocate.
  ImagePitch<T> currentLevel(0, 0, 0);
  for (int level = 0; level < this->numLevels; level++) {
    currentLevel = hostLevels[level];
    currentLevel.free();
  }
  check(cudaFree(deviceLevels));

  // Frees the temporary memory for hostLevels.
  check(cudaFreeHost(hostLevels));
}

template <typename T>
void PyramidImagePitch<T>::populateTopLevel(const std::vector<InterfaceImage<T>> &images,
                                            const int fromChannel) {
  printf("hello you need to implement me!!!!\n");
}

template <typename T>
__device__ T *PyramidImagePitch<T>::at(const int row, const int col, const int level) {
  return deviceLevels[level].at(row, col);
}

template <typename T>
__device__ const T *PyramidImagePitch<T>::constAt(const int row, const int col,
                                                  const int level) const {
  return deviceLevels[level].constAt(row, col);
}

template struct PyramidImagePitch<int>;
template struct PyramidImagePitch<float>;

} /* namespace StyLitCUDA */

#include "PyramidImagePitch.cuh"

#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <vector>

namespace StyLitCUDA {

template <typename T>
PyramidImagePitch<T>::PyramidImagePitch(const int rows, const int cols, const int numChannels,
                                          const int numLevels)
    : PyramidImage<T>(rows, cols, numChannels, numLevels), deviceData(nullptr),
      devicePitch(nullptr), deviceDimensions(nullptr) {}

template <typename T> void PyramidImagePitch<T>::allocate() {
  // Allocates device memory for deviceData, devicePitch and deviceDimensions.
  check(cudaMalloc(&deviceData, this->numLevels * sizeof(T *)));
  check(cudaMalloc(&devicePitch, this->numLevels * sizeof(size_t)));
  check(cudaMalloc(&deviceDimensions, this->numLevels * sizeof(Coordinates)));

  // Populates temporary copies of deviceData, devicePitch and deviceDimensions on the host.
  std::vector<T *> hostData(this->numLevels);
  std::vector<size_t> hostPitch(this->numLevels);
  std::vector<Coordinates> hostDimensions(this->numLevels);
  hostDimensions[0] = Coordinates(this->rows, this->cols);
  for (int level = 0; level < this->numLevels; level++) {
    if (level > 0) {
      hostDimensions[level] = hostDimensions[level - 1] / 2;
    }
    check(cudaMallocPitch(&hostData[level], &hostPitch[level],
                          this->numChannels * hostDimensions[level].col * sizeof(T),
                          hostDimensions[level].row));
  }

  // Copies hostData, hostPitch and hostDimensions to the device.
  check(cudaMemcpy((void *) deviceData, (void *) hostData.data(), this->numLevels * sizeof(T *), cudaMemcpyHostToDevice));
  check(cudaMemcpy((void *) devicePitch, (void *) hostPitch.data(), this->numLevels * sizeof(size_t), cudaMemcpyHostToDevice));
  check(cudaMemcpy((void *) deviceDimensions, (void *) hostDimensions.data(), this->numLevels * sizeof(Coordinates), cudaMemcpyHostToDevice));
}

template <typename T> void PyramidImagePitch<T>::free() {
  // Temporarily allocates memory for deviceData on the host.
  T **hostData;
  check(cudaMallocHost(&hostData, this->numLevels * sizeof(T *)));

  // Copies deviceData to the host.
  check(cudaMemcpy((void *) hostData, (void *) deviceData, this->numLevels * sizeof(T *), cudaMemcpyDeviceToHost));

  // Frees the images in each pyramid level.
  for (int level = 0; level < this->numLevels; level++) {
    check(cudaFree((void *) hostData[level]));
  }

  // Frees deviceData, devicePitch and deviceDimensions.
  check(cudaFree(deviceData));
  check(cudaFree(devicePitch));
  check(cudaFree(deviceDimensions));

  // Frees the temporarily allocated hostData.
  check(cudaFreeHost(hostData));
}

template <typename T>
__device__ const T *PyramidImagePitch<T>::at(const int row, const int col, const int level) {
  // T *image = deviceData[level];
  // T *rowStart = (T *)((char *)image + row * devicePitch[level]);
  // return &rowStart[col * this->numChannels];
  return nullptr;
}

template struct PyramidImagePitch<int>;
template struct PyramidImagePitch<float>;

} /* namespace StyLitCUDA */

#include "Vec.cuh"

#include "Utilities.cuh"

#include <cuda_runtime.h>

namespace StyLitCUDA {

template <typename T> Vec<T>::Vec(int size) : size(size) {}

template <typename T> void Vec<T>::deviceAllocate() {
  check(cudaMalloc(&deviceData, size * sizeof(T)));
}

template <typename T> void Vec<T>::deviceFree() {
  check(cudaFree(deviceData));
  deviceData = nullptr;
}

template <typename T> void Vec<T>::toDevice(T *data) {
  check(cudaMemcpy(deviceData, data, size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> void Vec<T>::fromDevice(T *data) {
  check(cudaMemcpy(data, deviceData, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template struct Vec<float>;

} /* namespace StyLitCUDA */

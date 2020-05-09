#include "Image.cuh"

#include "Utilities.cuh"
#include "../Interface/InterfaceImage.h"

#include <cuda_runtime.h>

namespace StyLitCUDA {

template <typename T>
Image<T>::Image(const int rows, const int cols, const int numChannels)
    : rows(rows), cols(cols), numChannels(numChannels) {}

template <typename T> void Image<T>::allocate() {
  check(cudaMallocPitch(&deviceData, &pitch, numChannels * cols * sizeof(T), rows));
}

template <typename T> void Image<T>::free() { check(cudaFree((void *) deviceData)); }

template <typename T>
bool Image<T>::populate(const std::vector<InterfaceImage<T>> &images) {
  // Verifies that the images are correctly sized.
  int numSuppliedChannels = 0;
  for (const InterfaceImage<T> &image : images) {
    numSuppliedChannels += image.numChannels;
    if (image.cols != cols || image.rows != rows) {
      return false;
    }
  }
  if (numSuppliedChannels > numChannels) {
    return false;
  }

  // Temporarily allocates space for the image on the host.
  T *hostImage;
  const int hostImageSizeInBytes = rows * cols * numChannels * sizeof(T);
  check(cudaMallocHost(&hostImage, hostImageSizeInBytes));
  memset(hostImage, 0, hostImageSizeInBytes);

  // Copies the images to hostImage.
  // Any unfilled channels in hostImage are zeroed out.
  int channel = 0;
  for (const InterfaceImage<T> &image : images) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        const T *interfaceImageVector = image.constAt(row, col);
        T *hostImageVector = &hostImage[numChannels * (row * cols + col)];
        for (int i = 0; i < image.numChannels; i++) {
          hostImageVector[channel + i] = interfaceImageVector[i];
        }
        channel += image.numChannels;
      }
    }
  }

  // Copies hostImage to the device and frees it.
  const int hostPitch = numChannels * cols * sizeof(T);
  check(cudaMemcpy2D((void *) deviceData, pitch, hostImage, hostPitch, hostPitch, rows, cudaMemcpyHostToDevice));
  check(cudaFreeHost(hostImage));
  return true;
}

template <typename T>
__device__ T *Image<T>::at(const int row, const int col) {
  T *rowStart = (T *)((char *)deviceData + row * pitch);
  return &rowStart[col * numChannels];
}

template <typename T>
__device__ const T *Image<T>::constAt(const int row, const int col) const {
  T *rowStart = (T *)((char *)deviceData + row * pitch);
  return &rowStart[col * numChannels];
}

template struct Image<int>;
template struct Image<float>;

} /* namespace StyLitCUDA */

#include "Image.cuh"

#include "../Algorithm/NNF.cuh"
#include "../Algorithm/PCG.cuh"
#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {

template <typename T>
Image<T>::Image(const int rows, const int cols, const int numChannels)
    : rows(rows), cols(cols), numChannels(numChannels) {}

template <typename T> void Image<T>::allocate() {
  check(cudaMallocPitch(&deviceData, &pitch, numChannels * cols * sizeof(T), rows));
}

template <typename T> void Image<T>::free() { check(cudaFree((void *)deviceData)); }

template <typename T> __device__ T *Image<T>::at(const int row, const int col) {
  T *rowStart = (T *)((char *)deviceData + row * pitch);
  return &rowStart[col * numChannels];
}

template <typename T> __device__ const T *Image<T>::constAt(const int row, const int col) const {
  T *rowStart = (T *)((char *)deviceData + row * pitch);
  return &rowStart[col * numChannels];
}

template <typename T>
int Image<T>::populateChannels(const std::vector<InterfaceImage<T>> &images,
                               const int fromChannel) {
  // Temporarily allocates space for the image on the host.
  T *hostData;
  const int hostImageSizeInBytes = rows * cols * numChannels * sizeof(T);
  check(cudaMallocHost(&hostData, hostImageSizeInBytes));
  memset(hostData, 0, hostImageSizeInBytes);

  // Copies the images to hostImage.
  // Any unfilled channels in hostImage are zeroed out.
  int channel = fromChannel;
  for (const InterfaceImage<T> &image : images) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        const T *interfaceImageVector = image.constAt(row, col);
        T *hostImageVector = &hostData[numChannels * (row * cols + col)];
        for (int i = 0; i < image.numChannels; i++) {
          hostImageVector[channel + i] = interfaceImageVector[i];
        }
      }
    }
    channel += image.numChannels;
  }

  // Copies hostImage to the device and frees it.
  const int hostPitch = numChannels * cols * sizeof(T);
  check(cudaMemcpy2D((void *)deviceData, pitch, hostData, hostPitch, hostPitch, rows,
                     cudaMemcpyHostToDevice));
  check(cudaFreeHost(hostData));

  // Returns the number of channels populated.
  return channel - fromChannel;
}

template <typename T>
int Image<T>::retrieveChannels(std::vector<InterfaceImage<T>> &images, const int fromChannel) {
  // Temporarily allocates space for the image on the host.
  T *hostData;
  const int hostImageSizeInBytes = rows * cols * numChannels * sizeof(T);
  check(cudaMallocHost(&hostData, hostImageSizeInBytes));

  // Copies deviceData to hostImage.
  const int hostPitch = numChannels * cols * sizeof(T);
  check(cudaMemcpy2D((void *)hostData, hostPitch, deviceData, pitch, hostPitch, rows,
                     cudaMemcpyDeviceToHost));

  // Copies the data to images.
  // Any unfilled channels in hostData are zeroed out.
  int channel = fromChannel;
  for (InterfaceImage<T> &image : images) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        T *interfaceImageVector = image.at(row, col);
        const T *hostImageVector = &hostData[numChannels * (row * cols + col)];
        for (int i = 0; i < image.numChannels; i++) {
          interfaceImageVector[i] = hostImageVector[channel + i];
        }
      }
    }
    channel += image.numChannels;
  }

  // Frees the temporary host space.
  check(cudaFreeHost(hostData));
  return channel - fromChannel;
}

template struct Image<int>;
template struct Image<float>;
template struct Image<NNFEntry>;
template struct Image<PCGState>;

} /* namespace StyLitCUDA */

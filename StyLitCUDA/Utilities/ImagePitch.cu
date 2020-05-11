#include "ImagePitch.cuh"

#include "Utilities.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace StyLitCUDA {

template <typename T>
ImagePitch<T>::ImagePitch(const int rows, const int cols, const int numChannels)
    : rows(rows), cols(cols), numChannels(numChannels) {}

template <typename T> void ImagePitch<T>::allocate() {
  fprintf(stderr, "check 1\n");
  check(
      cudaMallocPitch(&deviceData, &pitch, this->numChannels * this->cols * sizeof(T), this->rows));
  fprintf(stderr, "check 2\n");
}

template <typename T> void ImagePitch<T>::free() { check(cudaFree((void *)deviceData)); }

template <typename T> __device__ T *ImagePitch<T>::at(const int row, const int col) {
  T *rowStart = (T *)((char *)deviceData + row * pitch);
  return &rowStart[col * this->numChannels];
}

template <typename T>
__device__ const T *ImagePitch<T>::constAt(const int row, const int col) const {
  T *rowStart = (T *)((char *)deviceData + row * pitch);
  return &rowStart[col * this->numChannels];
}

template <typename T>
int ImagePitch<T>::populateChannels(const std::vector<InterfaceImage<T>> &images,
                                    const int fromChannel) {
  // Temporarily allocates space for the image on the host.
  T *hostImage;
  const int hostImageSizeInBytes = this->rows * this->cols * this->numChannels * sizeof(T);
  check(cudaMallocHost(&hostImage, hostImageSizeInBytes));
  memset(hostImage, 0, hostImageSizeInBytes);

  // Copies the images to hostImage.
  // Any unfilled channels in hostImage are zeroed out.
  int channel = fromChannel;
  for (const InterfaceImage<T> &image : images) {
    for (int row = 0; row < this->rows; row++) {
      for (int col = 0; col < this->cols; col++) {
        const T *interfaceImageVector = image.constAt(row, col);
        T *hostImageVector = &hostImage[this->numChannels * (row * this->cols + col)];
        for (int i = 0; i < image.numChannels; i++) {
          hostImageVector[channel + i] = interfaceImageVector[i];
        }
      }
    }
    channel += image.numChannels;
  }

  // Copies hostImage to the device and frees it.
  const int hostPitch = this->numChannels * this->cols * sizeof(T);
  check(cudaMemcpy2D((void *)deviceData, pitch, hostImage, hostPitch, hostPitch, this->rows,
                     cudaMemcpyHostToDevice));
  check(cudaFreeHost(hostImage));

  // Returns the number of channels populated.
  return channel - fromChannel;
}

template struct ImagePitch<int>;
template struct ImagePitch<float>;

} /* namespace StyLitCUDA */

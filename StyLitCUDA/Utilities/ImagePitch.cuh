#ifndef IMAGEPITCH_H_
#define IMAGEPITCH_H_

#include "Image.cuh"

namespace StyLitCUDA {

template <typename T> struct ImagePitch : public Image<T> {
public:
  ImagePitch(const int rows, const int cols, const int numChannels);
  virtual ~ImagePitch() = default;

  void allocate() override;

  void free() override;

  __device__ T *at(const int row, const int col) override;

  __device__ const T *constAt(const int row, const int col) const override;

  /**
   * @brief populateChannels Copies the specified images to deviceData. The InterfaceImages'
   * dimensions must match this ImagePitch's dimensions, and the sum of the InterfaceImages'
   * numChannels plus fromChannel must be less than or equal to this ImagePitch's numChannels.
   * @param images the images to copy to deviceData
   * @param fromChannel the channel in the ImagePitch to start from when copying
   * @return the number of channels populated
   */
  int populateChannels(const std::vector<InterfaceImage<T>> &images, const int fromChannel);

private:
  // the pitch returned by cudaMallocPitch
  size_t pitch = 0;

  // A device pointer to the image data (row major). The image data is not
  // initialized and freed in the constructor and destructor respectively
  // because ImagePitch is copied to the device during kernel launches.
  T *deviceData = nullptr;
};

} /* namespace StyLitCUDA */

#endif /* IMAGEPITCH_H_ */

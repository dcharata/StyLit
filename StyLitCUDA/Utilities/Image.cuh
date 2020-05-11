#ifndef IMAGEPITCH_H_
#define IMAGEPITCH_H_

#include "../Interface/InterfaceImage.h"

#include <vector>

namespace StyLitCUDA {

template <typename T> class Image {
public:
  Image(const int rows, const int cols, const int numChannels);
  virtual ~Image() = default;

  /**
   * @brief allocate Allocates the on-device memory associated with this
   * Image. This isn't in Image constructor because it should only be
   * triggered intentionally by the host, not when an Image is passed to the
   * device via a kernel.
   */
  void allocate();

  /**
   * @brief free Frees the on-device memory associated with this Image. This
   * isn't in Image's destructor because it should only be triggered
   * intentionally by the host, not when an Image is passed to the device
   * via a kernel.
   */
  void free();

  /**
   * @brief at Returns a pointer to the feature vector for the given coordinates.
   * @param row the row
   * @param col the column
   * @return a pointer to the feature vector for the given coordinates
   */
  __device__ T *at(const int row, const int col);

  /**
   * @brief constAt Returns a const pointer to the feature vector for the given coordinates.
   * @param row the row
   * @param col the column
   * @return a const pointer to the feature vector for the given coordinates
   */
  __device__ const T *constAt(const int row, const int col) const;

  /**
   * @brief populateChannels Copies the specified images to deviceData. The InterfaceImages'
   * dimensions must match this Image's dimensions, and the sum of the InterfaceImages'
   * numChannels plus fromChannel must be less than or equal to this Image's numChannels.
   * @param images the images to copy to deviceData
   * @param fromChannel the channel in the Image to start from when copying
   * @return the number of channels populated
   */
  int populateChannels(const std::vector<InterfaceImage<T>> &images, const int fromChannel);

  /**
   * @brief retrieveChannels Copies the specified images from deviceData. The InterfaceImages'
   * dimensions must match this Image's dimensions, and the sum of the InterfaceImages'
   * numChannels plus fromChannel must be less than or equal to this Image's numChannels.
   * @param images the image to copy into from deviceData
   * @param fromChannel the channel in the Image to start from when copying
   * @return the number of channels populated
   */
  int retrieveChannels(std::vector<InterfaceImage<T>> &images, const int fromChannel);

  // the number of rows in the image
  int rows;

  // the number of columns in the image
  int cols;

  // the number of channels in the image
  int numChannels;

  // the pitch returned by cudaMallocPitch
  size_t pitch = 0;

  // A device pointer to the image data (row major). The image data is not
  // initialized and freed in the constructor and destructor respectively
  // because Image is copied to the device during kernel launches.
  T *deviceData = nullptr;
};

} /* namespace StyLitCUDA */

#endif /* IMAGEPITCH_H_ */

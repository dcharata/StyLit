#ifndef IMAGE_H_
#define IMAGE_H_

#include "../Interface/InterfaceImage.h"

#include <vector>

namespace StyLitCUDA {

template <typename T> struct Image {
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
   * @brief populate Copies the specified images to deviceData. The
   * InterfaceImages' dimensions must match this Image's dimensions, and the sum
   * of the InterfaceImages' numChannels must be less than or equal to this
   * Image's numChannels.
   * @param images the images to copy to deviceData
   * @return true if successful; otherwise false
   */
  bool populate(const std::vector<InterfaceImage<T>> &images);

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

  // the number of rows in the image
  const int rows;

  // the number of columns in the image
  const int cols;

  // the number of channels in the image
  const int numChannels;

  // the pitch returned by cudaMallocPitch
  size_t pitch = 0;

  // A device pointer to the image data (row major). The image data is not
  // initialized and freed in the constructor and destructor respectively
  // because Image is copied to the device during kernel launches.
  const T *deviceData = nullptr;
};

} /* namespace StyLitCUDA */

#endif /* IMAGE_H_ */

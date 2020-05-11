#ifndef IMAGE_H_
#define IMAGE_H_

#include "../Interface/InterfaceImage.h"

#include <vector>

namespace StyLitCUDA {

template <typename T> struct Image {
public:
  Image(const int rows, const int cols, const int numChannels);
  virtual ~Image() = default;

  /**
   * @brief allocate Allocates the on-device memory associated with this
   * Image. This isn't in Image constructor because it should only be
   * triggered intentionally by the host, not when an Image is passed to the
   * device via a kernel.
   */
  virtual void allocate() = 0;

  /**
   * @brief free Frees the on-device memory associated with this Image. This
   * isn't in Image's destructor because it should only be triggered
   * intentionally by the host, not when an Image is passed to the device
   * via a kernel.
   */
  virtual void free() = 0;

  /**
   * @brief at Returns a pointer to the feature vector for the given coordinates.
   * @param row the row
   * @param col the column
   * @return a pointer to the feature vector for the given coordinates
   */
  __device__ virtual T *at(const int row, const int col) = 0;

  /**
   * @brief constAt Returns a const pointer to the feature vector for the given coordinates.
   * @param row the row
   * @param col the column
   * @return a const pointer to the feature vector for the given coordinates
   */
  __device__ virtual const T *constAt(const int row, const int col) const = 0;

  // the number of rows in the image
  int rows;

  // the number of columns in the image
  int cols;

  // the number of channels in the image
  int numChannels;
};

} /* namespace StyLitCUDA */

#endif /* IMAGE_H_ */

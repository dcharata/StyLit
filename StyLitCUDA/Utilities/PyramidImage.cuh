#ifndef READONLYIMAGE_H_
#define READONLYIMAGE_H_

#include "../Interface/InterfaceImage.h"

#include <vector>

namespace StyLitCUDA {

template <typename T> class PyramidImage {
public:
  PyramidImage(const int rows, const int cols, const int numChannels, const int numLevels);
  virtual ~PyramidImage() = default;

  virtual void allocate() = 0;

  virtual void free() = 0;

  /**
   * @brief populateTopLevel Populates the top level of the image with the given InterfaceImage.
   * @param images the InterfaceImages to populate the top pyramid level with
   * @param fromChannel the channel to start populating the image from
   */
  virtual void populateTopLevel(const std::vector<InterfaceImage<T>> &images,
                                const int fromChannel) = 0;

  __device__ virtual T *at(const int row, const int col, const int level) = 0;

  __device__ virtual T const *constAt(const int row, const int col, const int level) const = 0;

  const int rows;
  const int cols;
  const int numChannels;
  const int numLevels;
};

} /* namespace StyLitCUDA */

#endif /* READONLYIMAGE_H_ */

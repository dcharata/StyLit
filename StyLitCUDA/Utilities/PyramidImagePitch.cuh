#ifndef READONLYIMAGEPITCH_H_
#define READONLYIMAGEPITCH_H_

#include "../Interface/InterfaceImage.h"
#include "Coordinates.h"
#include "ImagePitch.cuh"

#include <vector>

namespace StyLitCUDA {

template <typename T> class PyramidImagePitch {
public:
  PyramidImagePitch(const int rows, const int cols, const int numChannels, const int numLevels);
  virtual ~PyramidImagePitch() = default;

  void allocate();

  void free();

  /**
   * @brief populateTopLevel Populates the top level of the image with the given InterfaceImage.
   * @param images the InterfaceImages to populate the top pyramid level with
   * @param fromChannel the channel to start populating the image from
   */
  void populateTopLevel(const std::vector<InterfaceImage<T>> &images,
                        const int fromChannel);

  __device__ T *at(const int row, const int col, const int level);

  __device__ T const *constAt(const int row, const int col, const int level) const;

  int rows;
  int cols;
  int numChannels;
  int numLevels;

private:
  ImagePitch<T> *deviceLevels;
};

} /* namespace StyLitCUDA */

#endif /* READONLYIMAGEPITCH_H_ */

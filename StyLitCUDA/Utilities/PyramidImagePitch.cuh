#ifndef READONLYIMAGEPITCH_H_
#define READONLYIMAGEPITCH_H_

#include "Coordinates.h"
#include "PyramidImage.cuh"

namespace StyLitCUDA {

template <typename T> class PyramidImagePitch : public PyramidImage<T> {
public:
  PyramidImagePitch(const int rows, const int cols, const int numChannels, const int numLevels);
  virtual ~PyramidImagePitch() = default;

  void allocate() override;

  void free() override;

  __device__ virtual const T *at(const int row, const int col, const int level) override;

private:
  // a device pointer to an array of T * (the images)
  T **deviceData;

  // a device pointer to an array of size_t (the pitch from allocation)
  size_t *devicePitch;

  // a device pointer to an array of image dimensions
  Coordinates *deviceDimensions;
};

} /* namespace StyLitCUDA */

#endif /* READONLYIMAGEPITCH_H_ */

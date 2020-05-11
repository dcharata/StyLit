#ifndef READONLYIMAGEPITCH_H_
#define READONLYIMAGEPITCH_H_

#include "Coordinates.h"
#include "ImagePitch.cuh"
#include "PyramidImage.cuh"

namespace StyLitCUDA {

template <typename T> class PyramidImagePitch : public PyramidImage<T> {
public:
  PyramidImagePitch(const int rows, const int cols, const int numChannels, const int numLevels);
  virtual ~PyramidImagePitch() = default;

  void allocate() override;

  void free() override;

  void populateTopLevel(const std::vector<InterfaceImage<T>> &images,
                        const int fromChannel) override;

  __device__ T *at(const int row, const int col, const int level) override;

  __device__ T const *constAt(const int row, const int col, const int level) const override;

private:
  ImagePitch<T> *deviceLevels;
};

} /* namespace StyLitCUDA */

#endif /* READONLYIMAGEPITCH_H_ */

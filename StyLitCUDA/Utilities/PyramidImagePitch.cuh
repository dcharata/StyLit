#ifndef READONLYIMAGEPITCH_H_
#define READONLYIMAGEPITCH_H_

#include "../Interface/InterfaceImage.h"
#include "Coordinates.h"
#include "ImagePitch.cuh"

#include <vector>

namespace StyLitCUDA {

template <typename T> struct PyramidImagePitch {
  PyramidImagePitch(const int rows, const int cols, const int numChannels, const int numLevels);
  virtual ~PyramidImagePitch();
  int rows;
  int cols;
  int numChannels;
  int numLevels;
  std::vector<ImagePitch<T>> levels;
};

} /* namespace StyLitCUDA */

#endif /* READONLYIMAGEPITCH_H_ */

#ifndef READONLYIMAGEPITCH_H_
#define READONLYIMAGEPITCH_H_

#include "../Interface/InterfaceImage.h"
#include "Image.cuh"

#include <vector>

namespace StyLitCUDA {

template <typename T> struct PyramidImage {
  PyramidImage(const int rows, const int cols, const int numChannels, const int numLevels);
  virtual ~PyramidImage();
  int rows;
  int cols;
  int numChannels;
  int numLevels;
  std::vector<Image<T>> levels;
};

} /* namespace StyLitCUDA */

#endif /* READONLYIMAGEPITCH_H_ */

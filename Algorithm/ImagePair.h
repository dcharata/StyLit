#ifndef IMAGEPAIR_H
#define IMAGEPAIR_H

#include "Image.h"

/**
 * @brief The ImagePair struct This defines an image pair.
 * The image pairs used in StyLit are the guides (A/B) and the styles (A'/B').
 */
template <typename T, unsigned int numChannels> struct ImagePair {
  ImagePair(ImageDimensions sourceDims, ImageDimensions targetDims) : source(sourceDims), target(targetDims)
  {
  }

  // the source image (e.g. A or A')
  Image<T, numChannels> source;

  // the target image (e.g. B or B')
  Image<T, numChannels> target;
};

#endif // IMAGEPAIR_H

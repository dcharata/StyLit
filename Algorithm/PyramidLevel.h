#ifndef PYRAMIDLEVEL_H
#define PYRAMIDLEVEL_H

#include "ImagePair.h"
#include "NNF.h"

/**
 * @brief The PyramidLevel struct This holds the information that's needed
 * across pyramid levels during StyLit.
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
struct PyramidLevel {
  PyramidLevel(ImageDimensions sourceDims, ImageDimensions targetDims)
      : forwardNNF(targetDims, sourceDims), reverseNNF(sourceDims, targetDims)
  {
  }

  ImagePair<T, numGuideChannels> guide;
  ImagePair<T, numStyleChannels> style;
  NNF forwardNNF; // this is the target sized array of source indices
  NNF reverseNNF; // this is the source sized array of target indices
};

#endif // PYRAMIDLEVEL_H

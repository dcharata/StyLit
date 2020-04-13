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
  ImagePair<T, numGuideChannels> guide;
  ImagePair<T, numStyleChannels> style;
  NNF forwardNNF;
  NNF reverseNNF;
};

#endif // PYRAMIDLEVEL_H

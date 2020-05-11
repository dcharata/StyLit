#ifndef PYRAMID_H
#define PYRAMID_H

#include "ChannelWeights.h"
#include "PyramidLevel.h"

#include <vector>

/**
 * @brief The Pyramid struct This holds the image pyramid used for StyLit and
 * its weights.
 */
template <typename T, unsigned int numGuideChannels, unsigned int numStyleChannels> struct Pyramid {
  std::vector<PyramidLevel<T, numGuideChannels, numStyleChannels>> levels;
  ChannelWeights<numGuideChannels> guideWeights;
  ChannelWeights<numStyleChannels> styleWeights;
};

#endif // PYRAMID_H

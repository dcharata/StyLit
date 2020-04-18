#ifndef ERRORCALCULATOR_H
#define ERRORCALCULATOR_H

#include "ImageDimensions.h"
#include "PyramidLevel.h"
#include "ChannelWeights.h"

struct Configuration;

/**
 * @brief The ErrorCalculator class This is the implementation-specific error
 * calculator. It takes a pyramid level and set of coordinates and calculates
 * the error.
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class ErrorCalculator {
public:
  ErrorCalculator() = default;
  virtual ~ErrorCalculator() = default;

  /**
   * @brief calculateError This is a wrapper around
   * implementationOfCalculateError. It checks that the source and target
   * coordinates are within bounds.
   * @param configuration the configuration StyLit is running
   * @param pyramidLevel the pyramid level to calculate an error for
   * @param sourceCoordinates the coordinates in the source image
   * @param targetCoordinates the coordinates in the target image
   * @param error the out argument for the error
   * @return true if calculating the error succeeds; otherwise false
   */
  bool calculateError(
      const Configuration &configuration,
      const PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel,
      const ImageCoordinates &sourceCoordinates,
      const ImageCoordinates &targetCoordinates, float &error) {
    Q_ASSERT(pyramidLevel.guide.source.dimensions ==
             pyramidLevel.style.source.dimensions);
    Q_ASSERT(pyramidLevel.guide.target.dimensions ==
             pyramidLevel.style.target.dimensions);
    Q_ASSERT(sourceCoordinates.within(pyramidLevel.guide.source.dimensions));
    Q_ASSERT(targetCoordinates.within(pyramidLevel.guide.target.dimensions));
    return implementationOfCalculateError(configuration, pyramidLevel,
                                          sourceCoordinates, targetCoordinates,
                                          error);
  }

protected:
  /**
   * @brief implementationOfCalculateError Calculates the error between the
   * source and target for the given coordinates and pyramid level.
   * @param configuration the configuration StyLit is running
   * @param pyramidLevel the pyramid level to calculate an error for
   * @param sourceCoordinates the coordinates in the source image
   * @param targetCoordinates the coordinates in the target image
   * @param error the out argument for the error
   * @return true if calculating the error succeeds; otherwise false
   */
  virtual bool implementationOfCalculateError(
      const Configuration &configuration,
      const PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel,
      const ImageCoordinates &sourceCoordinates,
      const ImageCoordinates &targetCoordinates,
      const ChannelWeights<numGuideChannels> &guideWeights,
      const ChannelWeights<numStyleChannels> &styleWeights, float &error) = 0;
};

#endif // ERRORCALCULATOR_H

#ifndef ERRORCALCULATORCPU_H
#define ERRORCALCULATORCPU_H

#include "Algorithm/ErrorCalculator.h"
#include "Algorithm/ChannelWeights.h"
#include "Configuration/Configuration.h"
#include <iostream>

/**
 * @brief The ErrorCalculator class This is the implementation-specific error
 * calculator. It takes a pyramid level and set of coordinates and calculates
 * the error.
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class ErrorCalculatorCPU : public ErrorCalculator<T, numGuideChannels, numStyleChannels> {
public:
  ErrorCalculatorCPU() = default;
  virtual ~ErrorCalculatorCPU() = default;

private:

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
  virtual bool implementationOfCalculateError(const Configuration &configuration,
                                              const PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel,
                                              const ImageCoordinates &sourceCoordinates,
                                              const ImageCoordinates &targetCoordinates,
                                              const ChannelWeights<numGuideChannels> &guideWeights,
                                              const ChannelWeights<numStyleChannels> &styleWeights, float &error) {
    error = 0;
    ImageCoordinates min = ImageCoordinates{0,0};
    ImageCoordinates A_max = pyramidLevel.guide.source.dimensions;
    ImageCoordinates B_max = pyramidLevel.guide.target.dimensions;
    const int centerRowSource = sourceCoordinates.row;
    const int centerColSource = sourceCoordinates.col;
    const int centerRowTarget = targetCoordinates.row;
    const int centerColTarget = targetCoordinates.col;

    const int PATCH_SIZE = configuration.patchSize;

    // go through all of the feature vectors from the guide and style in the patch and subtract the
    // source from the target. Add (source - target)^2 * weight to the total error.
    for (int colOffset = -PATCH_SIZE / 2; colOffset <= PATCH_SIZE / 2; colOffset++) {
      for (int rowOffset = -PATCH_SIZE / 2; rowOffset <= PATCH_SIZE / 2; rowOffset++) {
        ImageCoordinates sourceCoords = ImageCoordinates{qBound(min.row, centerRowSource + rowOffset, A_max.row - 1),
                                                         qBound(min.col, centerColSource + colOffset, A_max.col - 1)};
        ImageCoordinates targetCoords = ImageCoordinates{qBound(min.row, centerRowTarget + rowOffset, B_max.row - 1),
                                                         qBound(min.col, centerColTarget + colOffset, B_max.col - 1)};
        FeatureVector<T, numGuideChannels> guideDiff = pyramidLevel.guide.source.getConstPixel(sourceCoords.row, sourceCoords.col) // A
                                                       - pyramidLevel.guide.target.getConstPixel(targetCoords.row, targetCoords.col); // B
        error += (guideDiff.array().square() * guideWeights.array()).matrix().sum();
        FeatureVector<T, numStyleChannels> styleDiff = pyramidLevel.style.source.getConstPixel(sourceCoords.row, sourceCoords.col) // A'
                                                       - pyramidLevel.style.target.getConstPixel(targetCoords.row, targetCoords.col); // B'
        error += (styleDiff.array().square() * styleWeights.array()).matrix().sum();
      }
    }

    return true;
  }
};

#endif // ERRORCALCULATORCPU_H

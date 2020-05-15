#ifndef ERRORCALCULATORCPU_H
#define ERRORCALCULATORCPU_H

#include "Algorithm/ChannelWeights.h"
#include "Algorithm/ErrorCalculator.h"
#include "Configuration/Configuration.h"
#include <iostream>

/**
 * @brief The ErrorCalculator class This is the implementation-specific error
 * calculator. It takes a pyramid level and set of coordinates and calculates
 * the error.
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class ErrorCalculatorCPU
    : public ErrorCalculator<T, numGuideChannels, numStyleChannels> {
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
  virtual bool implementationOfCalculateError(
      const Configuration &configuration,
      const PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel,
      const ImageCoordinates &sourceCoordinates,
      const ImageCoordinates &targetCoordinates,
      const ChannelWeights<numGuideChannels> &guideWeights,
      const ChannelWeights<numStyleChannels> &styleWeights, float &error) {
    error = 0;
    ImageCoordinates sourceDims = pyramidLevel.guide.source.dimensions;
    ImageCoordinates targetDims = pyramidLevel.guide.target.dimensions;

    const int PATCH_SIZE = configuration.patchSize;
    const int HALF_PATCH_SIZE = configuration.patchSize / 2;

    const int rowSourceBegin = sourceCoordinates.row - HALF_PATCH_SIZE;
    const int colSourceBegin = sourceCoordinates.col - HALF_PATCH_SIZE;
    const int rowTargetBegin = targetCoordinates.row - HALF_PATCH_SIZE;
    const int colTargetBegin = targetCoordinates.col - HALF_PATCH_SIZE;

    FeatureVector<T, numGuideChannels> *sourceGuidePtr = pyramidLevel.guide.source.data.get() + (rowSourceBegin * sourceDims.cols + colSourceBegin);
    FeatureVector<T, numStyleChannels> *sourceStylePtr = pyramidLevel.style.source.data.get() + (rowSourceBegin * sourceDims.cols + colSourceBegin);
    FeatureVector<T, numGuideChannels> *targetGuidePtr = pyramidLevel.guide.target.data.get() + (rowTargetBegin * targetDims.cols + colTargetBegin);
    FeatureVector<T, numStyleChannels> *targetStylePtr = pyramidLevel.style.target.data.get() + (rowTargetBegin * targetDims.cols + colTargetBegin);

    int sourceWidthOffset = sourceDims.cols - PATCH_SIZE;
    int targetWidthOffset = targetDims.cols - PATCH_SIZE;

    // go through all of the feature vectors from the guide and style in the
    // patch and subtract the source from the target. Add (source - target)^2 *
    // weight to the total error.
    if (sourceCoordinates.patchWithin(sourceDims, HALF_PATCH_SIZE) &&
        targetCoordinates.patchWithin(targetDims, HALF_PATCH_SIZE)) {
      for (int i = 0; i < PATCH_SIZE; i++) {
        for (int j = 0; j < PATCH_SIZE; j++) {
          FeatureVector<T, numGuideChannels> guideDiff = *(sourceGuidePtr) - *(targetGuidePtr);
          error += guideWeights.cwiseProduct(guideDiff.cwiseProduct(guideDiff)).sum();
          //error += guideDiff.cwiseProduct(guideDiff).sum();

          FeatureVector<T, numStyleChannels> styleDiff = *(sourceStylePtr) - *(targetStylePtr);

          error += styleWeights.cwiseProduct(styleDiff.cwiseProduct(styleDiff)).sum();
          //error += styleDiff.cwiseProduct(styleDiff).sum();

          sourceGuidePtr++;
          sourceStylePtr++;
          targetGuidePtr++;
          targetStylePtr++;
        }
        sourceGuidePtr += sourceWidthOffset;
        sourceStylePtr += sourceWidthOffset;
        targetGuidePtr += targetWidthOffset;
        targetStylePtr += targetWidthOffset;
      }
    } else {
      for (int rowOffset = -PATCH_SIZE / 2; rowOffset <= PATCH_SIZE / 2;
           rowOffset++) {
        for (int colOffset = -PATCH_SIZE / 2; colOffset <= PATCH_SIZE / 2;
             colOffset++) {
          ImageCoordinates sourceCoords = ImageCoordinates{
              qBound(0, sourceCoordinates.row + rowOffset, sourceDims.rows - 1),
              qBound(0, sourceCoordinates.col + colOffset, sourceDims.cols - 1)};
          ImageCoordinates targetCoords = ImageCoordinates{
              qBound(0, targetCoordinates.row + rowOffset, targetDims.rows - 1),
              qBound(0, targetCoordinates.col + colOffset, targetDims.cols - 1)};
          FeatureVector<T, numGuideChannels> guideDiff =
              pyramidLevel.guide.source.getConstPixel(sourceCoords.row,
                                                      sourceCoords.col) // A
              - pyramidLevel.guide.target.getConstPixel(targetCoords.row,
                                                        targetCoords.col); // B
          error += guideWeights.cwiseProduct(guideDiff.cwiseProduct(guideDiff)).sum();
          FeatureVector<T, numStyleChannels> styleDiff =
              pyramidLevel.style.source.getConstPixel(sourceCoords.row,
                                                      sourceCoords.col) // A'
              - pyramidLevel.style.target.getConstPixel(targetCoords.row,
                                                        targetCoords.col); // B'

          error += styleWeights.cwiseProduct(styleDiff.cwiseProduct(styleDiff)).sum();
        }
      }
    }
    return true;
  }
};

#endif // ERRORCALCULATORCPU_H

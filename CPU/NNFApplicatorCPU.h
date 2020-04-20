#ifndef NNFAPPLICATORCPU_H
#define NNFAPPLICATORCPU_H

#include "Algorithm/NNFApplicator.h"
#include <iostream>

struct Configuration;

/**
 * @brief The NNFApplicator class This is the interface for the
 * implementation-specified NNF applicator. In StyLit, it generates a stylized
 * target image by averaging patches in a pyramid level's NNF and storing
 * the results in the level's stylized target image.
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class NNFApplicatorCPU : public NNFApplicator<T, numGuideChannels, numStyleChannels> {
public:
  NNFApplicatorCPU() = default;
  virtual ~NNFApplicatorCPU() = default;

private:

  const int PATCH_SIZE = 5;

  /**
   * @brief implementationOfApplyNNF Generates a stylized target image by
   *        averaging the forward NNF of the pyramid level
   * @param configuration the configuration StyLit is running
   * @param pyramidLevel the pyramidLevel with a filled forward NNF for which
   *        we are generating the stylized target image
   * @return true if stylized target image generation succeeds; otherwise false
   */
   bool implementationOfApplyNNF(const Configuration &configuration,
                                 PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel) {
     int targetCols = pyramidLevel.style.target.dimensions.cols;
     int targetRows = pyramidLevel.style.target.dimensions.rows;
     for (int col = 0; col < targetCols; col++) {
       for (int row = 0; row < targetRows; row++) {
         // create a final pixel value and keep track of the weight added
         FeatureVector<T, numStyleChannels> finalPix;
         for (int i = 0; i < numStyleChannels; i++) {
           finalPix(i) = 0;
         }
         float sumWeight = 0;
         for (int colOffset = -PATCH_SIZE / 2; colOffset <= PATCH_SIZE / 2; colOffset++) {
           for (int rowOffset = -PATCH_SIZE / 2; rowOffset <= PATCH_SIZE / 2; rowOffset++) {
             ImageCoordinates offsetTargetCoords{row + rowOffset, col + colOffset};
             // only add to the average if the pixel we are looking at actually exists in the target
             if (offsetTargetCoords.within(pyramidLevel.style.target.dimensions)) {
               // get the source coordinates from the NNF and then offset them as shown in the ebsynth source
               ImageCoordinates sourceCoords = pyramidLevel.forwardNNF.getMapping(offsetTargetCoords);
               ImageCoordinates offsetSourceCoords{sourceCoords.row - rowOffset, sourceCoords.col - colOffset};
               // only add to the average if the pixel we are looking at actually exists in the source
               if (offsetSourceCoords.within(pyramidLevel.style.source.dimensions)) {
                 finalPix += pyramidLevel.style.source(offsetSourceCoords.row, offsetSourceCoords.col);
                 sumWeight += 1.0;
               }
             }
           }
         }
         pyramidLevel.style.target.data[row * targetCols + col] = finalPix / sumWeight;
       }
     }
  }
};

#endif // NNFAPPLICATOR_H

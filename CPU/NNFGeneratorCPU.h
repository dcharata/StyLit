#ifndef NNFGENERATORCPU_H
#define NNFGENERATORCPU_H

#include "Algorithm/FeatureVector.h"
#include "Algorithm/NNFError.h"
#include "Algorithm/NNFGenerator.h"
#include "Algorithm/PyramidLevel.h"
#include "ErrorBudgetCalculatorCPU.h"
#include "ErrorCalculatorCPU.h"
#include "PatchMatcherCPU.h"
#include <iostream>

struct Configuration;

/**
 * @brief The NNFGeneratorCPU class creates a foward NNF for
 * one iteration of Algorithm 1 in the Stylit paper
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class NNFGeneratorCPU
    : public NNFGenerator<T, numGuideChannels, numStyleChannels> {
public:
  NNFGeneratorCPU() = default;
  ~NNFGeneratorCPU() = default;

private:
  float NNF_GENERATION_STOPPING_CRITERION = 0.95f;

  /**
   * @brief implementationOfGenerateNNF Generates a forward NNF by repeatedly
   * sampling and updating a reverse NNF. The forward NNF in the PyramidLevel
   * should be updated. This might end up needed the next-coarsest PyramidLevel
   * as an argument as well. Currently, the reverse NNF is always initialized
   * randomly in the first iteration.
   * @param configuration the configuration StyLit is running
   * @param pyramid the image pyramid. The forward nnf of this level should be
   *        initialized.
   * @param level the level of the pyramid for which the forward NNF is being
   * generated
   * @return true if NNF generation succeeds; otherwise false
   */
  bool implementationOfGenerateNNF(
      const Configuration &configuration,
      Pyramid<T, numGuideChannels, numStyleChannels> &pyramid, int level) {
    PatchMatcherCPU<T, numGuideChannels, numStyleChannels> patchMatcher =
        PatchMatcherCPU<T, numGuideChannels, numStyleChannels>();
    PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel =
        pyramid.levels[level];
    ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels> errorCalc =
        ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels>();
    ErrorBudgetCalculatorCPU budgetCalc = ErrorBudgetCalculatorCPU();

    // create and initialize the blacklist
    NNF blacklist = NNF(pyramidLevel.guide.target.dimensions,
                        pyramidLevel.guide.source.dimensions);
    blacklist.setToInitializedBlacklist();

    // the source dimensions of the forward NNF are the dimesions of the target
    int patchesFilled = 0;
    bool firstIteration = true;
    const int forwardNNFSize = pyramidLevel.forwardNNF.sourceDimensions.area();
    int numIterations = 0;
    while (patchesFilled <
           float(forwardNNFSize) * NNF_GENERATION_STOPPING_CRITERION) {

      std::cout << "*************************" << std::endl;
      std::cout << "Fraction of patches filled: " << patchesFilled << " / "
                << float(forwardNNFSize) << std::endl;
      std::cout << "*************************" << std::endl;

      if (firstIteration) {
        patchMatcher.patchMatch(configuration, pyramidLevel.reverseNNF, pyramid,
                                configuration.numPatchMatchIterations, level,
                                true, true, nullptr);
        firstIteration = false;
      } else {
        patchMatcher.patchMatch(configuration, pyramidLevel.reverseNNF, pyramid,
                                configuration.numPatchMatchIterations, level,
                                true, true, &blacklist);
      }

      // fill up the error image of the nnfError struct
      NNFError nnfError(pyramidLevel.reverseNNF);
      float totalError = 0;
      int invalidMappings = 0;
      for (int col = 0; col < nnfError.nnf.sourceDimensions.cols; col++) {
        for (int row = 0; row < nnfError.nnf.sourceDimensions.rows; row++) {
          Q_ASSERT(
              (ImageDimensions{row, col}).within(nnfError.error.dimensions));
          ImageCoordinates blacklistVal = blacklist.getMapping(
              pyramidLevel.reverseNNF.getMapping(ImageDimensions{row, col}));
          if (blacklistVal == ImageCoordinates::FREE_PATCH) { // we only need to add the errors of valid mappings to the total error
            float patchError = 0;
            ImageCoordinates currentPatch{row, col};
            errorCalc.calculateError(configuration, pyramidLevel, currentPatch,
                                     nnfError.nnf.getMapping(currentPatch),
                                     pyramid.guideWeights, pyramid.styleWeights,
                                     patchError);
            nnfError.error(row, col) = FeatureVector<float, 1>(patchError);

            totalError += patchError;
          } else { // if the mapping is invalid, just fill the error image with max float
            nnfError.error(row, col) = FeatureVector<float, 1>(std::numeric_limits<float>::max());
            invalidMappings++;
          }
        }
      }
      std::cout << "Total error: " << totalError << std::endl;
      std::cout << "Invalid Mappings: " << invalidMappings << std::endl;

      // get the error budget
      float budget;
      std::vector<std::pair<int, float>> sortedCoordinates;
      budgetCalc.calculateErrorBudget(configuration, sortedCoordinates, nnfError, totalError, budget, &blacklist);
      //budgetCalc.calculateErrorBudget(configuration, sortedCoordinates, nnfError, totalError, budget, nullptr);

      std::cout << "Budget: " << budget << std::endl;

      // fill up the forward NNF using the reverse NNF until we hit the error
      // budget
      float pastError = 0;
      int i = 0;
      const int width = nnfError.nnf.sourceDimensions.cols;
      int notFreeCount = 0;
      int numAddedToForwardNNFInIteration = 0;
      int recentlyTakenCount = 0;
      int redundantBackgroundMatches = 0;
      int redundantObjectMatches = 0;
      while (pastError < budget && i < int(sortedCoordinates.size())) {
        ImageCoordinates coords{sortedCoordinates[i].first / width,
                                sortedCoordinates[i].first % width};
        // if coords does not map to a blacklisted pixel, then we can create
        // this mapping in the forward NNF
        ImageCoordinates blacklistVal =
            blacklist.getMapping(pyramidLevel.reverseNNF.getMapping(coords));
        if (blacklistVal == ImageCoordinates::FREE_PATCH) {
          pyramidLevel.forwardNNF.setMapping(
              pyramidLevel.reverseNNF.getMapping(coords), coords);
          // record which iteration this target patch was added to blacklist
          blacklist.setMapping(pyramidLevel.reverseNNF.getMapping(coords),
                               ImageCoordinates{numIterations, numIterations});
          pastError = sortedCoordinates[i].second;
          numAddedToForwardNNFInIteration++;
          patchesFilled++;
        } else if (blacklistVal == ImageCoordinates{numIterations, numIterations}) {
          recentlyTakenCount++;
          if (pyramidLevel.style.source(coords.row, coords.col)(0,0) > pyramidLevel.style.source(coords.row, coords.col)(2,0)) {
            redundantBackgroundMatches++;
          } else {
            redundantObjectMatches++;
          }
        } else {
          notFreeCount++;
        }
        i++;
      }
      std::cout << "Final past error: " << pastError << std::endl;
      std::cout << "Not free count: " << notFreeCount << std::endl;
      std::cout << "Not free because taken in this iteration: " << recentlyTakenCount << std::endl;
      std::cout << "Number of patches added to forward NNF in this iteration: " << numAddedToForwardNNFInIteration << std::endl;
      std::cout << "Redundant background matches: " << redundantBackgroundMatches << std::endl;
      std::cout << "Redundant object matches: " << redundantObjectMatches << std::endl;
      std::cout << "i = " << i << " is out of " << sortedCoordinates.size()
                << std::endl;
      numIterations++;
    }

    // if the level's forward NNf is not completely full, make a new forward NNF
    // from patchmatch and use that to fill up the holes in the level's forward
    // NNF
    if (patchesFilled < forwardNNFSize) {
      NNF forwardNNFFinal = NNF(pyramidLevel.guide.target.dimensions,
                                pyramidLevel.guide.source.dimensions);
      patchMatcher.patchMatch(configuration, forwardNNFFinal, pyramid,
                              configuration.numPatchMatchIterations, level,
                              false, true);
      for (int col = 0; col < forwardNNFFinal.sourceDimensions.cols; col++) {
        for (int row = 0; row < forwardNNFFinal.sourceDimensions.rows; row++) {
          ImageCoordinates currentPatch{row, col};
          ImageCoordinates blacklistVal = blacklist.getMapping(currentPatch);
          if (blacklistVal == ImageCoordinates::FREE_PATCH) {
            pyramidLevel.forwardNNF.setMapping(
                currentPatch, forwardNNFFinal.getMapping(currentPatch));
          }
        }
      }
    }

    return true;
  }

  /**
   * @brief alternateImplementationOfGenerateNNF Generates a forward NNF by
   * repeatedly sampling and updating another forward NNF. The forward NNF in
   * the PyramidLevel should be updated. This might end up needed the
   * next-coarsest PyramidLevel as an argument as well.
   * @param configuration the configuration StyLit is running
   * @param pyramid the image pyramid
   * @param level the level of the pyramid for which the forward NNF is being
   * generated
   * @return true if NNF generation succeeds; otherwise false
   */
  bool alternateImplementationOfGenerateNNF(
      const Configuration &, Pyramid<T, numGuideChannels, numStyleChannels> &,
      int) {
    return true;
  }
};

#endif // NNFGENERATORCPU_H

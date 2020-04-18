#ifndef NNFGENERATORCPU_H
#define NNFGENERATORCPU_H

#include "Algorithm/NNFGenerator.h"
#include "Algorithm/PyramidLevel.h"
#include "PatchMatcherCPU.h"
#include "Algorithm/NNFError.h"
#include "ErrorCalculatorCPU.h"
#include "Algorithm/FeatureVector.h"
#include "ErrorBudgetCalculatorCPU.h"

struct Configuration;

/**
 * @brief The NNFGeneratorCPU class creates a foward NNF for
 * one iteration of Algorithm 1 in the Stylit paper
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class NNFGeneratorCPU : NNFGenerator<T, numGuideChannels, numStyleChannels> {
public:
  NNFGeneratorCPU() = default;
  ~NNFGeneratorCPU() = default;

private:

  float NNF_GENERATION_STOPPING_CRITERION = .95;

  /**
   * @brief implementationOfGenerateNNF Generates a forward NNF by repeatedly
   * sampling and updating a reverse NNF. The forward NNF in the PyramidLevel
   * should be updated. This might end up needed the next-coarsest PyramidLevel
   * as an argument as well. Currently, the reverse NNF is always initialized
   * randomly in the first iteration.
   * @param configuration the configuration StyLit is running
   * @param pyramid the image pyramid. The forward nnf of this level should be
   *        initialized.
   * @param level the level of the pyramid for which the forward NNF is being generated
   * @return true if NNF generation succeeds; otherwise false
   */
  bool implementationOfGenerateNNF(const Configuration &configuration,
                                   Pyramid<T, numGuideChannels, numStyleChannels> &pyramid, int level) {

    PatchMatcherCPU<T, numGuideChannels, numStyleChannels> patchMatcher = PatchMatcherCPU<T, numGuideChannels, numStyleChannels>();
    PyramidLevel<T, numGuideChannels, numStyleChannels> pyramidLevel = pyramid.levels[level];
    ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels> errorCalc = ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels>();
    ErrorBudgetCalculatorCPU budgetCalc = ErrorBudgetCalculatorCPU();

    // create and initialize the blacklist
    NNF blacklist = NNF(pyramidLevel.guide.target.dimensions, pyramidLevel.guide.source.dimensions);
    blacklist.setToInitializedBlacklist();

    // the source dimensions of the forward NNF are the dimesions of the target
    int patchesFilled = 0;
    bool firstIteration = 0;
    int forwardNNFSize = pyramidLevel.forwardNNF.sourceDimensions.rows * pyramidLevel.forwardNNF.sourceDimensions.cols;

    while (patchesFilled < float(forwardNNFSize) * NNF_GENERATION_STOPPING_CRITERION) {
      if (firstIteration) {
        patchMatcher.patchMatch(configuration, pyramidLevel.reverseNNF, pyramid, level, true, true, nullptr);
        firstIteration = false;
      } else {
        patchMatcher.patchMatch(configuration, pyramidLevel.reverseNNF, pyramid, level, true, true, blacklist);
      }

      // fill up the error image of the nnfError struct
      NNFError nnfError(pyramidLevel.reverseNNF);
      for (int col = 0; col < nnfError.nnf.sourceDimensions.cols; col++) {
        for (int row = 0; row < nnfError.nnf.sourceDimensions.cols; row++) {
          // not sure how the error image is initialized, so I added this so that there won't be any out of bounds errors
          assert(nnfError.error.dimensions.within(ImageDimensions{row, col}));
          float patchError;
          ImageCoordinates currentPatch{row, col};
          patchError = errorCalc.calculateError(configuration, pyramidLevel, currentPatch, nnfError.nnf.getMapping(currentPatch), patchError);
          nnfError.error.data[row * nnfError.nnf.sourceDimensions.cols + col] = FeatureVector<float, 1>(patchError);
        }
      }

      // get the error budget
      float budget;
      std::vector<ImageCoordinates> sortedCoordinates; // **************** these need to be sorted by calculateErrorBudget ************************
      budgetCalc.calculateErrorBudget(configuration, nnfError, budget);

      // fill up the forward NNF using the reverse NNF until we hit the error budget
      float errorIntegral = 0;
      int i = 0;
      while (errorIntegral < budget) {
        pyramidLevel.forwardNNF.setMapping(pyramidLevel.reverseNNF.getMapping(sortedCoordinates[i]), sortedCoordinates[i]);
        blacklist.setMapping(pyramidLevel.reverseNNF.getMapping(sortedCoordinates[i]), sortedCoordinates[i]);
        i++;
      }
    }

    // if the level's forward NNf is not completely full, make a new forward NNF from patchmatch and
    // use that to fill up the holes in the level's forward NNF
    if (patchesFilled < forwardNNFSize) {
      NNF forwardNNF = NNF(pyramidLevel.guide.target.dimensions, pyramidLevel.guide.source.dimensions);
      patchMatcher.patchMatch(configuration, forwardNNF, pyramid, level, false, true);
      for (int col = 0; col < forwardNNF.sourceDimensions.cols; col++) {
        for (int row = 0; row < forwardNNF.sourceDimensions.rows; row++) {
          ImageCoordinates currentPatch{row, col};
          ImageCoordinates blacklistVal = blacklist.getMapping(currentPatch);
          if (blacklistVal.col == -1 && blacklistVal.row == -1) {
            pyramidLevel.forwardNNF.setMapping(forwardNNF.getMapping(currentPatch));
          }
        }
      }
    }

    return true;
  }

  /**
   * @brief alternateImplementationOfGenerateNNF Generates a forward NNF by repeatedly
   * sampling and updating another forward NNF. The forward NNF in the PyramidLevel
   * should be updated. This might end up needed the next-coarsest PyramidLevel
   * as an argument as well.
   * @param configuration the configuration StyLit is running
   * @param pyramid the image pyramid
   * @param level the level of the pyramid for which the forward NNF is being generated
   * @return true if NNF generation succeeds; otherwise false
   */
   bool alternateImplementationOfGenerateNNF(const Configuration &configuration,
                                             Pyramid<T, numGuideChannels, numStyleChannels> &pyramid, int level) {
     return true;
   }
};

#endif // NNFGENERATORCPU_H

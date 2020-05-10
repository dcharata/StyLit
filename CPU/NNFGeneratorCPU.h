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

bool generatorComparator(const std::pair<float, ImageCoordinates> lhs,
                const std::pair<float, ImageCoordinates> rhs) {
  return lhs.first < rhs.first;
}

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
  const int MAX_ITERATIONS = 6;

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
      Pyramid<T, numGuideChannels, numStyleChannels> &pyramid, int level,
      std::vector<float> &budgets) {
    PatchMatcherCPU<T, numGuideChannels, numStyleChannels> patchMatcher =
        PatchMatcherCPU<T, numGuideChannels, numStyleChannels>();
    PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel =
        pyramid.levels[level];
    ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels> errorCalc =
        ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels>();
    ErrorBudgetCalculatorCPU budgetCalc = ErrorBudgetCalculatorCPU();

    // figure out if we are in the first optimization iteration
    bool firstOptimizationIteration = budgets.size() == 0;

    // create and initialize the blacklist
    NNF blacklist = NNF(pyramidLevel.guide.target.dimensions,
                        pyramidLevel.guide.source.dimensions);
    blacklist.setToInitializedBlacklist();

    // the source dimensions of the forward NNF are the dimesions of the target
    int patchesFilled = 0;
    bool firstIteration = true;
    const int forwardNNFSize = pyramidLevel.forwardNNF.sourceDimensions.area();
    int iteration = 0;

    while (patchesFilled < float(forwardNNFSize) * NNF_GENERATION_STOPPING_CRITERION && iteration < MAX_ITERATIONS) {

      std::cout << "*************************" << std::endl;
      std::cout << "Fraction of patches filled: " << patchesFilled << " / "
                << float(forwardNNFSize) << std::endl;

      NNFError nnfError(pyramidLevel.reverseNNF);

      if (firstIteration) {
        patchMatcher.patchMatch(configuration, pyramidLevel.reverseNNF, pyramid,
                                level, true, true, &nnfError, nullptr);
        firstIteration = false;
      } else {
        patchMatcher.patchMatch(configuration, pyramidLevel.reverseNNF, pyramid,
                                level, true, true, &nnfError, &blacklist);
      }

      std::vector<std::pair<float, ImageCoordinates>> sortedCoordinates;
      float totalError = 0;
      getSortedCoordinates(sortedCoordinates, nnfError, totalError);

      // calculate the error budget if we are in the first optimization iteration of this pyramid level
      float budget = 0;
      if (firstOptimizationIteration) {
        budgetCalc.calculateErrorBudget(configuration, sortedCoordinates, nnfError, totalError, budget, &blacklist);
        budgets.push_back(budget);
      } else {
        budget = budgets[std::min<int>(iteration, budgets.size() - 1)];
      }

      std::cout << "Budget: " << budget << std::endl;

      // fill up the forward NNF using the reverse NNF until we hit the error
      // budget
      float pastError = 0;
      int i = 0;
      int notFreeCount = 0;
      int numAddedToForwardNNFInIteration = 0;
      int recentlyTakenCount = 0;
      while (pastError < budget && i < int(sortedCoordinates.size())) {
        ImageCoordinates coords = sortedCoordinates[i].second;
        // if coords does not map to a blacklisted pixel, then we can create
        // this mapping in the forward NNF
        ImageCoordinates blacklistVal =
            blacklist.getMapping(pyramidLevel.reverseNNF.getMapping(coords));
        if (blacklistVal == ImageCoordinates::FREE_PATCH) {
          pyramidLevel.forwardNNF.setMapping(
              pyramidLevel.reverseNNF.getMapping(coords), coords);
          // record which iteration this target patch was added to blacklist
          blacklist.setMapping(pyramidLevel.reverseNNF.getMapping(coords),
                               ImageCoordinates{iteration, iteration});
          pastError = sortedCoordinates[i].first;
          numAddedToForwardNNFInIteration++;
          patchesFilled++;
        } else if (blacklistVal == ImageCoordinates{iteration, iteration}) {
          recentlyTakenCount++;
        } else {
          notFreeCount++;
        }
        i++;
      }

      iteration++;
    }

    // if the level's forward NNf is not completely full, make a new forward NNF
    // from patchmatch and use that to fill up the holes in the level's forward
    // NNF
    if (patchesFilled < forwardNNFSize) {
      NNF forwardNNFFinal = NNF(pyramidLevel.guide.target.dimensions,
                                pyramidLevel.guide.source.dimensions);
      patchMatcher.patchMatch(configuration, forwardNNFFinal, pyramid,
                              level, false, true);
      for (int row = 0; row < forwardNNFFinal.sourceDimensions.rows; row++) {
        for (int col = 0; col < forwardNNFFinal.sourceDimensions.cols; col++) {
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

  void getSortedCoordinates(std::vector<std::pair<float, ImageCoordinates>> &sortedCoordinates,
                            NNFError &nnfError, float &totalError) {
    totalError = 0;
    for (int row = 0; row < nnfError.nnf.sourceDimensions.rows; row++) {
      for (int col = 0; col < nnfError.nnf.sourceDimensions.cols; col++) {
        if (nnfError.error(row, col)(0,0) < std::numeric_limits<float>::max()) {
          sortedCoordinates.push_back(std::pair(nnfError.error(row, col)(0,0), ImageCoordinates{row, col}));
          totalError += nnfError.error(row,col)(0,0);
        }
      }
    }
    std::sort(sortedCoordinates.begin(), sortedCoordinates.end(), &generatorComparator);
  }

};

#endif // NNFGENERATORCPU_H

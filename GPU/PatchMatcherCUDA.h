#ifndef PATCHMATCHERCUDA_H
#define PATCHMATCHERCUDA_H

#include "Algorithm/ChannelWeights.h"
#include "Algorithm/FeatureVector.h"
#include "Algorithm/NNF.h"
#include "Algorithm/PatchMatcher.h"
#include "Algorithm/PyramidLevel.h"
#include "CPU/ErrorCalculatorCPU.h"
#include "InputPatchMatcherCUDA.h"

#include <QtGlobal>
#include <iostream>
#include <limits>

template <typename T> void patchMatchCUDA(InputPatchMatcherCUDA<T> &);

template <typename T, unsigned int numGuideChannels, unsigned int numStyleChannels>
class PatchMatcherCUDA : public PatchMatcher<T, numGuideChannels, numStyleChannels> {
public:
  PatchMatcherCUDA() = default;
  ~PatchMatcherCUDA() = default;

  void randomlyInitializeNNF(NNF &nnf) {
    for (int col = 0; col < nnf.sourceDimensions.cols; col++) {
      for (int row = 0; row < nnf.sourceDimensions.rows; row++) {
        ImageCoordinates from{row, col};
        ImageCoordinates to{randi(0, nnf.targetDimensions.rows),
                            randi(0, nnf.targetDimensions.col)};
        nnf.setMapping(from, to);
      }
    }
  }

  int randi(int min, int max) { return (std::rand() % (max - min)) + min; }

private:
  bool implementationOfPatchMatch(const Configuration &, NNF &nnf,
                                  const Pyramid<T, numGuideChannels, numStyleChannels> &pyramid,
                                  int, int level, bool makeReverseNNF, bool,
                                  const NNF *const blacklist = nullptr) {
    // Makes sure the NNF matches makeReverseNNF.
    const PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel = pyramid.levels[level];
    if (makeReverseNNF) {
      Q_ASSERT(nnf.sourceDimensions == pyramidLevel.guide.source.dimensions);
    } else {
      Q_ASSERT(nnf.sourceDimensions == pyramidLevel.guide.target.dimensions);
    }

    // Extracts the data that CUDA will need.
    InputPatchMatcherCUDA<T> input;
    input.sourceRows = nnf.sourceDimensions.rows;
    input.sourceCols = nnf.sourceDimensions.cols;
    input.targetRows = nnf.targetDimensions.rows;
    input.targetCols = nnf.targetDimensions.cols;
    input.numGuideChannels = numGuideChannels;
    input.numStyleChannels = numStyleChannels;
    input.hostNNF = nnf.getData();
    input.hostBlacklist = blacklist;

    // Extracts the source and target images.
    const T *a = (T *)pyramidLevel.guide.source.data.get();
    const T *b = (T *)pyramidLevel.guide.target.data.get();
    const T *aPrime = (T *)pyramidLevel.style.source.data.get();
    const T *bPrime = (T *)pyramidLevel.style.target.data.get();
    input.hostGuideSource = makeReverseNNF ? a : b;
    input.hostGuideTarget = makeReverseNNF ? b : a;
    input.hostStyleSource = makeReverseNNF ? aPrime : bPrime;
    input.hostStyleTarget = makeReverseNNF ? bPrime : aPrime;

    patchMatchCUDA<T>(input);

    return true;
  }
};

#endif // PATCHMATCHERCUDA_H

#ifndef PATCHMATCHERCPU_H
#define PATCHMATCHERCPU_H

#include "Algorithm/PatchMatcher.h"
#include "Algorithm/FeatureVector.h"
#include "Algorithm/PyramidLevel.h"
#include "Algorithm/ChannelWeights.h"
#include "Algorithm/NNFError.h"
#include "ErrorCalculatorCPU.h"
#include <iostream>
#include <limits>

class NNF;

/**
 * @brief Implements the PatchMatch algorithm on the CPU
 */

template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class PatchMatcherCPU
    : public PatchMatcher<T, numGuideChannels, numStyleChannels> {
public:
  PatchMatcherCPU() = default;
  ~PatchMatcherCPU() = default;

  void randomlyInitializeNNF(NNF &nnf) {
    for (int row = 0; row < nnf.sourceDimensions.rows; row++) {
      for (int col = 0; col < nnf.sourceDimensions.cols; col++) {
        ImageCoordinates from{ row, col };
        ImageCoordinates to{ randi(0, nnf.targetDimensions.rows),
                             randi(0, nnf.targetDimensions.col) };
        nnf.setMapping(from, to);
      }
    }
  }

  void initNNFError(NNFError &nnfError) {
    for (int row = 0; row < nnfError.nnf.sourceDimensions.rows; row++) {
      for (int col = 0; col < nnfError.nnf.sourceDimensions.cols; col++) {
        nnfError.error(row, col) = FeatureVector<float, 1>(BIG_ERROR);
      }
    }
  }

  void initNNFErrorProperly(
      const Configuration &configuration, NNFError &nnfError, const NNF &nnf,
      bool makeReverseNNF,
      const PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel,
      const ChannelWeights<numGuideChannels> &guideWeights,
      const ChannelWeights<numStyleChannels> &styleWeights,
      ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels> &calc) {
    for (int row = 0; row < nnfError.nnf.sourceDimensions.rows; row++) {
      for (int col = 0; col < nnfError.nnf.sourceDimensions.cols; col++) {
        float error = 0;
        if (makeReverseNNF) {
          ImageCoordinates coords{ row, col };
          calc.calculateError(configuration, pyramidLevel, coords,
                              nnf.getMapping(coords), guideWeights,
                              styleWeights, error);
        } else {
          ImageCoordinates coords{ row, col };
          calc.calculateError(configuration, pyramidLevel,
                              nnf.getMapping(coords), coords, guideWeights,
                              styleWeights, error);
        }
        nnfError.error(row, col) = FeatureVector<float, 1>(error);
      }
    }
  }

  void initOmega(const Configuration &configuration, std::vector<float> &omega,
                 ImageDimensions dims, int PATCH_SIZE) {
    omega.assign(dims.rows * dims.cols, 0);
    for (int row = 0; row < dims.rows; row++) {
      for (int col = 0; col < dims.cols; col++) {
        updateOmegaValue(configuration, omega, row, col, dims, PATCH_SIZE, 1);
      }
    }
  }

private:
  const float RANDOM_SEARCH_ALPHA = .5;
  const float BIG_ERROR = 100000;

  /**
   * @brief patchMatch This is a wrapper around implementationOfPatchMatch. It
   * currently doesn't do any error checks, but I included it so that
   * PatchMatcher's format is the same as that of Downscaler, NNFUpscaler, etc.
   * @param configuration the configuration StyLit is running
   * @param nnf the NNF that should be improved with PatchMatch, size of domain
   * images,
   *        maps to indices in codomain images
   * @param pyramid the image pyramid
   * @param numIterations number of iterations that patchmatch is run
   * @param level the level of the pyramid for which an NNF is being generated
   * @param makeReverseNNF indicates whether a reverse or forward nnf is
   *        being generated
   * @param initRandom indicates whether the NNF should have patchmatch run on
   * it as is, or
   *        whether it should be randomly initialized
   * @param blacklist Another NNF of pixels that should not be mapped to.
   * @return true if patch matching succeeds; otherwise false
   */
  bool implementationOfPatchMatch(
      const Configuration &configuration, NNF &nnf,
      const Pyramid<T, numGuideChannels, numStyleChannels> &pyramid, int level,
      bool makeReverseNNF, bool initRandom, NNFError &nnfError, bool initError,
      std::vector<float> &omega, const ImageDimensions omegaDimensions,
      NNF *const blacklist = nullptr) {

    const PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel =
        pyramid.levels[level];

    const int numNNFRows = nnf.sourceDimensions.rows;
    const int numNNFCols = nnf.sourceDimensions.cols;

    const ChannelWeights<numGuideChannels> guideWeights = pyramid.guideWeights;
    const ChannelWeights<numStyleChannels> styleWeights = pyramid.styleWeights;

    if (initRandom) {
      randomlyInitializeNNF(nnf);
    }

    if (initError) {
      // initNNFError(nnfError);
      ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels> errorCalc =
          ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels>();
      initNNFErrorProperly(configuration, nnfError, nnf, makeReverseNNF,
                           pyramidLevel, guideWeights, styleWeights, errorCalc);
    }

    // precompute the search step radii
    int i = 0;
    const int w = std::max(
        std::max(nnf.sourceDimensions.cols, nnf.sourceDimensions.rows),
        std::max(nnf.targetDimensions.cols, nnf.targetDimensions.rows));
    std::vector<int> radii;
    while (w * std::pow(RANDOM_SEARCH_ALPHA, i) > 1.0) {
      radii.push_back(w * std::pow(RANDOM_SEARCH_ALPHA, i));
      i++;
    }

    for (int i = 0; i < configuration.numPatchMatchIterations; i++) {
      bool iterationIsOdd = i % 2 == 1 ? true : false;
#pragma omp parallel for schedule(dynamic)
      for (int row = 0; row < numNNFRows; row++) {
        for (int col = 0; col < numNNFCols; col++) {
          propagationStep(configuration, row, col, makeReverseNNF,
                          iterationIsOdd, nnf, pyramidLevel, guideWeights,
                          styleWeights, nnfError, omega, omegaDimensions,
                          blacklist);
          searchStep(configuration, row, col, makeReverseNNF, nnf, pyramidLevel,
                     guideWeights, styleWeights, radii, nnfError, omega,
                     omegaDimensions, blacklist);
        }
      }
    }

    return true;
  }

  /**
   * @brief propagationStep This runs the propagation step of patchmatch. It
   * propagates
   * information throughout the NNF by having each element of the NNF consider
   * mapping
   * to a patch in the codomain that is close to the element's neighbor's
   * mappings.
   * @param configuration the configuration StyLit is running
   * @param row the row of the element of the NNF that is being mutated
   * @param col the col of the element of the NNF that is being mutated
   * @param makeReverseNNF indicates whether a reverse or forward nnf is
   *        being generated
   * @param iterationIsOdd indicates whether the iteration of patchmatch is odd
   * or not
   * @param nnf the NNF that should be improved with PatchMatch, size of domain
   * images,
   *        maps to indices in codomain images
   * @param pyramid the image pyramid
   * @param level the level of the pyramid for which an NNF is being generated
   * @param guideWeights the guideweights of the pyramid
   * @param styleWeights the styleweights of the pyramid
   * @param blacklist Another NNF of pixels that should not be mapped to.
   * @return true if patch matching succeeds; otherwise false
   */
  void propagationStep(
      const Configuration &configuration, int row, int col, bool makeReverseNNF,
      bool iterationIsOdd, NNF &nnf,
      const PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel,
      const ChannelWeights<numGuideChannels> &guideWeights,
      const ChannelWeights<numStyleChannels> &styleWeights, NNFError &nnfError,
      std::vector<float> &omega, const ImageDimensions omegaDimensions,
      const NNF *const blacklist = nullptr) {
    ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels> errorCalc =
        ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels>();
    float newPatchError1 = -1.0; // we know if a patch was out of bounds if its
                                 // error remains -1, so don't consider it the
                                 // end of this method
    float newPatchError2 = -1.0;
    int offset = iterationIsOdd ? -1 : 1;
    const ImageCoordinates currentPatch{ row, col };
    ImageCoordinates newPatch1{ -1, -1 };
    ImageCoordinates newPatch2{ -1, -1 };
    bool newPatch1Available = false;
    const ImageCoordinates domainNeighbor1{ row + offset, col };
    if (domainNeighbor1.within(nnf.sourceDimensions)) { // if the neighbor in
                                                        // the nnf domain
                                                        // actually exists
      ImageCoordinates codomainNeighbor1 = nnf.getMapping(domainNeighbor1);
      // get the patch in the codomain that we might want to map the current
      // (row, col) domain patch to
      // NOTE: the -offset below is from the ebsynth implementation, and is not
      // part of the original patchmatch algorithm
      // This modification seems to reduce blurriness when we average the
      // contents of the NNF into a final image
      // and makes the total error of the NNF decrease slightly faster
      newPatch1.row = codomainNeighbor1.row - offset;
      newPatch1.col = codomainNeighbor1.col;
      // the blacklist tells us if the codomain index newPatch1 is available.
      if (newPatch1.within(nnf.targetDimensions)) { // if new codomain patch is
                                                    // in the codomain
                                                    // dimensions
        // if the corresponding element in the blacklist is (-1,-1), then this
        // new patch is available
        newPatch1Available =
            (blacklist == nullptr) ||
            (blacklist->getMapping(newPatch1) == ImageCoordinates::FREE_PATCH);
        if (newPatch1Available) {
          if (makeReverseNNF) {
            errorCalc.calculateError(configuration, pyramidLevel, currentPatch,
                                     newPatch1, guideWeights, styleWeights,
                                     newPatchError1);
          } else {
            errorCalc.calculateError(configuration, pyramidLevel, newPatch1,
                                     currentPatch, guideWeights, styleWeights,
                                     newPatchError1);
          }
        } else {
          newPatchError1 =
              std::numeric_limits<float>::max(); // if newPatch1 is not
                                                 // available, automatically set
                                                 // the energy to MAXFLOAT
        }
      }
    }

    // do the exact same thing as above but with the analogous col offset
    bool newPatch2Available = false;
    const ImageCoordinates domainNeighbor2{ row, col + offset };
    if (domainNeighbor2.within(nnf.sourceDimensions)) { // if the neighbor in te
                                                        // nnf domain actually
                                                        // exists
      ImageCoordinates codomainNeighbor2 = nnf.getMapping(domainNeighbor2);
      // NOTE: we have the same -offset that we have above
      newPatch2.row = codomainNeighbor2.row;
      newPatch2.col = codomainNeighbor2.col - offset;
      if (newPatch2.within(nnf.targetDimensions)) { // if the new codomain patch
                                                    // is in the codomain
                                                    // dimensions
        newPatch2Available =
            (blacklist == nullptr) ||
            (blacklist->getMapping(newPatch2) == ImageCoordinates::FREE_PATCH);
        if (newPatch2Available) {
          if (makeReverseNNF) {
            errorCalc.calculateError(configuration, pyramidLevel, currentPatch,
                                     newPatch2, guideWeights, styleWeights,
                                     newPatchError2);
          } else {
            errorCalc.calculateError(configuration, pyramidLevel, newPatch2,
                                     currentPatch, guideWeights, styleWeights,
                                     newPatchError2);
          }
        } else {
          newPatchError2 = std::numeric_limits<float>::max();
        }
      }
    }

    // calculate the energy from the current mapping
    /*
    float currentError;
    if (makeReverseNNF) {
      errorCalc.calculateError(configuration, pyramidLevel, currentPatch,
    nnf.getMapping(currentPatch), guideWeights, styleWeights, currentError);
    } else {
      errorCalc.calculateError(configuration, pyramidLevel,
    nnf.getMapping(currentPatch), currentPatch, guideWeights, styleWeights,
    currentError);
    }
    */
    ImageCoordinates oldCodomainPatch = nnf.getMapping(currentPatch);
    int PATCH_SIZE = configuration.patchSize;
    float omegaError =
        nnfError.error(row, col)(0, 0) +
        computeOmegaValue(configuration, omega, oldCodomainPatch.row,
                          oldCodomainPatch.col, omegaDimensions, PATCH_SIZE);
    float newPatchOmegaError1 = 0;
    if (newPatch1Available) {
      newPatchOmegaError1 =
          newPatchError1 + computeOmegaValue(configuration, omega,
                                             newPatch1.row, newPatch1.col,
                                             omegaDimensions, PATCH_SIZE);
    }
    float newPatchOmegaError2 = 0;
    if (newPatch2Available) {
      newPatchOmegaError2 =
          newPatchError2 + computeOmegaValue(configuration, omega,
                                             newPatch2.row, newPatch2.col,
                                             omegaDimensions, PATCH_SIZE);
    }
    // now that we have the errors of the new patches we are considering and the
    // current error, we can decide which one is the best
    bool changedToNewPatch1 = false;
    if (newPatch1Available && newPatchOmegaError1 < omegaError) {
      updateOmegaValue(configuration, omega, oldCodomainPatch.row,
                       oldCodomainPatch.col, omegaDimensions, PATCH_SIZE, -1);
      updateOmegaValue(configuration, omega, newPatch1.row, newPatch1.col,
                       omegaDimensions, PATCH_SIZE, 1);
      nnf.setMapping(currentPatch, newPatch1);
      nnfError.error(row, col) = FeatureVector<float, 1>(newPatchError1);
      changedToNewPatch1 = true;
    }

    if (changedToNewPatch1) {
      if (newPatch2Available && newPatchOmegaError2 < newPatchOmegaError1) {
        updateOmegaValue(configuration, omega, newPatch1.row, newPatch1.col,
                         omegaDimensions, PATCH_SIZE, -1);
        updateOmegaValue(configuration, omega, newPatch2.row, newPatch2.col,
                         omegaDimensions, PATCH_SIZE, 1);
        nnf.setMapping(currentPatch, newPatch2);
        nnfError.error(row, col) = FeatureVector<float, 1>(newPatchError2);
      }
    } else {
      if (newPatch2Available && newPatchOmegaError2 < omegaError) {
        updateOmegaValue(configuration, omega, oldCodomainPatch.row,
                         oldCodomainPatch.col, omegaDimensions, PATCH_SIZE, -1);
        updateOmegaValue(configuration, omega, newPatch2.row, newPatch2.col,
                         omegaDimensions, PATCH_SIZE, 1);
        nnf.setMapping(currentPatch, newPatch2);
        nnfError.error(row, col) = FeatureVector<float, 1>(newPatchError2);
      }
    }
  }

  /**
   * @brief searchStep This runs the search step of patchmatch. It has an
   * element of the
   * NNF search for a better mapping by repeatedly randomly searching in a
   * radius of
   * decreasing size.
   * @param row the row of the element of the NNF that is being mutated
   * @param col the col of the element of the NNF that is being mutated
   * @param makeReverseNNF indicates whether a reverse or forward nnf is
   *        being generated
   * @param nnf the NNF that should be improved with PatchMatch, size of domain
   * images,
   *        maps to indices in codomain images
   * @param pyramid the image pyramid
   * @param level the level of the pyramid for which an NNF is being generated
   * @param guideWeights the guideweights of the pyramid
   * @param styleWeights the styleweights of the pyramid
   * @param blacklist Another NNF of pixels that should not be mapped to.
   * @return true if patch matching succeeds; otherwise false
   */
  void searchStep(
      const Configuration &configuration, int row, int col, bool makeReverseNNF,
      NNF &nnf,
      const PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel,
      const ChannelWeights<numGuideChannels> guideWeights,
      const ChannelWeights<numStyleChannels> styleWeights,
      const std::vector<int> &radii, NNFError &nnfError,
      std::vector<float> &omega, const ImageDimensions omegaDimensions,
      const NNF *const blacklist = nullptr) {
    // NOTE: maximum search radius is the largest dimension of the images. We
    // should tune this later on.
    ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels> errorCalc =
        ErrorCalculatorCPU<T, numGuideChannels, numStyleChannels>();
    const ImageCoordinates currentPatch{ row, col };
    ImageCoordinates currentCodomainPatch = nnf.getMapping(currentPatch);
    /*
    float currentError;
    if (makeReverseNNF) {
      errorCalc.calculateError(configuration, pyramidLevel, currentPatch,
    currentCodomainPatch, guideWeights, styleWeights, currentError);
    } else {
      errorCalc.calculateError(configuration, pyramidLevel,
    currentCodomainPatch, currentPatch, guideWeights, styleWeights,
    currentError);
    }
    */
    ImageCoordinates oldCodomainPatch = nnf.getMapping(currentPatch);
    int PATCH_SIZE = configuration.patchSize;
    float currentError = nnfError.error(row, col)(0, 0);
    float currentOmegaError =
        currentError +
        computeOmegaValue(configuration, omega, oldCodomainPatch.row,
                          oldCodomainPatch.col, omegaDimensions, PATCH_SIZE);
    // float currentError = nnfError.error(row, col)(0,0);
    for (int i = 0; i < radii.size(); i++) {
      const int col_offset = int(radii[i] * rand_uniform());
      const int row_offset = int(radii[i] * rand_uniform());
      const ImageCoordinates newCodomainPatch{
        currentCodomainPatch.row + row_offset,
        currentCodomainPatch.col + col_offset
      };
      if (newCodomainPatch.within(
              nnf.targetDimensions)) { // if this new codomain patch is within
                                       // the codomain dimensions of the nnf
        bool newCodomainPatchAvailable =
            (blacklist == nullptr) ||
            (blacklist->getMapping(newCodomainPatch) ==
             ImageCoordinates::FREE_PATCH);
        if (newCodomainPatchAvailable) { // it is only worth to check whether we
                                         // should move to this new patch if it
                                         // is not on the blacklist
          float newError;
          if (makeReverseNNF) {
            errorCalc.calculateError(configuration, pyramidLevel, currentPatch,
                                     newCodomainPatch, guideWeights,
                                     styleWeights, newError);
          } else {
            errorCalc.calculateError(configuration, pyramidLevel,
                                     newCodomainPatch, currentPatch,
                                     guideWeights, styleWeights, newError);
          }
          float newOmegaError =
              newError + computeOmegaValue(
                             configuration, omega, newCodomainPatch.row,
                             newCodomainPatch.col, omegaDimensions, PATCH_SIZE);
          if (newOmegaError < currentOmegaError) { // update the patch that
                                                   // currentPatch maps to if it
                                                   // has lower error
            /*
            if ((std::rand() % 10000) == 0) {
              std::cout << "new " << newError << " " << newOmegaError <<
            std::endl;
              std::cout << "curr " << currentError << " " << currentOmegaError
            << std::endl;
            }
            */
            nnf.setMapping(currentPatch, newCodomainPatch);
            nnfError.error(row, col) = FeatureVector<float, 1>(newError);
            updateOmegaValue(configuration, omega, currentCodomainPatch.row,
                             currentCodomainPatch.col, omegaDimensions,
                             PATCH_SIZE, -1);
            updateOmegaValue(configuration, omega, newCodomainPatch.row,
                             newCodomainPatch.col, omegaDimensions, PATCH_SIZE,
                             1);
            currentCodomainPatch.row = newCodomainPatch.row;
            currentCodomainPatch.col = newCodomainPatch.col;
            currentError = newError;
            currentOmegaError =
                currentError + computeOmegaValue(configuration, omega,
                                                 currentCodomainPatch.row,
                                                 currentCodomainPatch.col,
                                                 omegaDimensions, PATCH_SIZE);
          }
        }
      }
    }
  }

  inline float computeOmegaValue(const Configuration &configuration,
                                 const std::vector<float> &omega, int row,
                                 int col, ImageDimensions dims,
                                 int PATCH_SIZE) {
    if (configuration.omegaWeight <= 0) {
      return 0.0f;
    }
    float ret = 0;
    int HALF_PATCH_SIZE = PATCH_SIZE / 2;
    int PATCH_SIZE_SQUARED = PATCH_SIZE * PATCH_SIZE;
    float mult = configuration.omegaWeight / PATCH_SIZE_SQUARED;
    for (int rowOffset = -HALF_PATCH_SIZE; rowOffset <= HALF_PATCH_SIZE;
         rowOffset++) {
      for (int colOffset = -HALF_PATCH_SIZE; colOffset <= HALF_PATCH_SIZE;
           colOffset++) {
        if (ImageCoordinates { row + rowOffset, col + colOffset }.within(
                dims)) {
          ret += omega[(row + rowOffset) * dims.cols + (col + colOffset)];
        }
      }
    }
    return mult * ret;
  }

  inline void updateOmegaValue(const Configuration &configuration,
                               std::vector<float> &omega, int row, int col,
                               ImageDimensions dims, int PATCH_SIZE,
                               int change) {
    if (configuration.omegaWeight <= 0) {
      return;
    }
    int HALF_PATCH_SIZE = PATCH_SIZE / 2;
    for (int rowOffset = -HALF_PATCH_SIZE; rowOffset <= HALF_PATCH_SIZE;
         rowOffset++) {
      for (int colOffset = -HALF_PATCH_SIZE; colOffset <= HALF_PATCH_SIZE;
           colOffset++) {
        if (ImageCoordinates { row + rowOffset, col + colOffset }.within(
                dims)) {
          omega[(row + rowOffset) * dims.cols + (col + colOffset)] += change;
        }
      }
    }
  }

  inline int randi(int min, int max) {
    return (std::rand() % (max - min)) + min;
  }

  inline float rand_uniform() {
    return (float(std::rand()) / float(INT_MAX)) * 2.0 - 1.0;
  }
};

#endif // PATCHMATCHERCPU_H

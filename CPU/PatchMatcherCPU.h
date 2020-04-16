#ifndef PATCHMATCHERCPU_H
#define PATCHMATCHERCPU_H

#include "Algorithm/PatchMatcher.h"
#include "Algorithm/FeatureVector.h"
#include "Algorithm/PyramidLevel.h"
#include "Algorithm/ChannelWeights.h"

class NNF;

/**
 * @brief Implements the PatchMatch algorithm on the CPU
 */

template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class PatchMatcherCPU : public PatchMatcher<T, numGuideChannels, numStyleChannels> {
public:
  PatchMatcherCPU() = default;
  ~PatchMatcherCPU() = default;

private:
  /**
   * @brief patchMatch This is a wrapper around implementationOfPatchMatch. It
   * currently doesn't do any error checks, but I included it so that
   * PatchMatcher's format is the same as that of Downscaler, NNFUpscaler, etc.
   * @param configuration the configuration StyLit is running
   * @param nnf the NNF that should be improved with PatchMatch, size of domain images,
   *        maps to indices in codomain images
   * @param pyramid the image pyramid
   * @param level the level of the pyramid for which an NNF is being generated
   * @param make_reverse_nnf indicates whether a reverse or forward nnf is
   *        being generated
   * @param blacklist Another NNF of pixels that should not be mapped to. See
   * the comment for implementationOfPatchMatch for more details.
   * @return true if patch matching succeeds; otherwise false
   */
  virtual bool implementationOfPatchMatch(const Configuration &configuration, NNF &nnf,
                                          const Pyramid<T, numGuideChannels, numStyleChannels> &pyramid,
                                          int level, bool make_reverse_nnf, const NNF *const blacklist = nullptr) {


  }

};

#endif // PATCHMATCHERCPU_H

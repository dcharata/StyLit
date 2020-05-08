#ifndef PATCHMATCHER_H
#define PATCHMATCHER_H

#include "Algorithm/Pyramid.h"
#include "Configuration/Configuration.h"

class NNF;

/**
 * @brief The PatchMatcher class This is the interface for the
 * implementation-specific PatchMatch function, which is used in the creation of
 * NNFs (nearest-neighbor fields). The NNF must have the size of the domain and
 * will map to indices in the codomain.
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class PatchMatcher {
public:
  PatchMatcher() = default;
  virtual ~PatchMatcher() = default;

  /**
   * @brief patchMatch This is a wrapper around implementationOfPatchMatch. It
   * currently doesn't do any error checks, but I included it so that
   * PatchMatcher's format is the same as that of Downscaler, NNFUpscaler, etc.
   * @param configuration the configuration StyLit is running
   * @param nnf the NNF that should be improved with PatchMatch, size of domain
   * images, maps to indices in codomain images
   * @param pyramid the image pyramid
   * @param level the level of the pyramid for which an NNF is being generated
   * @param make_reverse_nnf indicates whether a reverse or forward nnf is
   *        being generated
   * @param nnf the NNF that should be improved with PatchMatch
   * @param blacklist Another NNF of pixels that should not be mapped to. See
   * the comment for implementationOfPatchMatch for more details.
   * @return true if patch matching succeeds; otherwise false
   */
  bool patchMatch(const Configuration &configuration, NNF &nnf,
                  const Pyramid<T, numGuideChannels, numStyleChannels> &pyramid,
                  int numIterations, int level, bool makeReverseNNF,
                  bool initRandom, const NNF *const blacklist = nullptr) {
    return implementationOfPatchMatch(configuration, nnf, pyramid,
                                      numIterations, level, makeReverseNNF,
                                      initRandom, blacklist);
  }

protected:
  /**
   * @brief patchMatch This is a wrapper around implementationOfPatchMatch. It
   * currently doesn't do any error checks, but I included it so that
   * PatchMatcher's format is the same as that of Downscaler, NNFUpscaler, etc.
   * @param configuration the configuration StyLit is running
   * @param nnf the NNF that should be improved with PatchMatch, size of domain
   * images, maps to indices in codomain images
   * @param pyramid the image pyramid
   * @param level the level of the pyramid for which an NNF is being generated
   * @param makeReverseNNF indicates whether a reverse or forward nnf is
   *        being generated const NNF *const blacklist = nullptr);
   * @param blacklist This optional NNF goes in the opposite direction of NNF.
   * Source coordinates that correspond to valid mappings in blacklist should
   * not be mapped to. This is used in the iterative creation of NNFs via
   * reverse NNFs in the knee point step, where parts of the forward NNF that
   * have already been mapped shouldn't be mapped again.
   * @return true if patch matching succeeds; otherwise false
   */
  virtual bool implementationOfPatchMatch(
      const Configuration &configuration, NNF &nnf,
      const Pyramid<T, numGuideChannels, numStyleChannels> &pyramid,
      int numIterations, int level, bool makeReverseNNF, bool initRandom,
      const NNF *const blacklist = nullptr) = 0;
};

#endif // PATCHMATCHER_H

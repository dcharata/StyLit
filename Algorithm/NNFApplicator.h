#ifndef NNFAPPLICATOR_H
#define NNFAPPLICATOR_H

#include "PyramidLevel.h"

struct Configuration;

/**
 * @brief The NNFApplicator class This is the interface for the
 * implementation-specified NNF applicator. In StyLit, it generates a stylized
 * target image by averaging patches in a pyramid level's NNF and storing
 * the results in the level's stylized target image.
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class NNFApplicator {
public:
  NNFApplicator() = default;
  virtual ~NNFApplicator() = default;

  /**
   * @brief applyNNF This is a wrapper around
   * implementationOfApplyNNF.
   * @param configuration the configuration StyLit is running
   * @param pyramidLevel the pyramidLevel with a filled forward NNF for which
   *        we are generating the stylized target image
   * @return true if stylized target image generation succeeds; otherwise false
   */
  bool applyNNF(const Configuration &configuration,
                PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel) {
    return implementationOfApplyNNF(configuration, pyramidLevel);
  }

protected:
  /**
   * @brief implementationOfApplyNNF Generates a stylized target image by
   *        averaging the forward NNF of the pyramid level
   * @param configuration the configuration StyLit is running
   * @param pyramidLevel the pyramidLevel with a filled forward NNF for which
   *        we are generating the stylized target image
   * @return true if stylized target image generation succeeds; otherwise false
   */
  virtual bool implementationOfApplyNNF(
      const Configuration &configuration,
      PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel) = 0;
};

#endif // NNFAPPLICATOR_H

#ifndef NNFGENERATOR_H
#define NNFGENERATOR_H

#include "PyramidLevel.h"

struct Configuration;

/**
 * @brief The NNFGenerator class This is the interface for the
 * implementation-specified NNF generator. In StyLit, it generates forward NNFs
 * by repeatedly sampling a reverse NNF, but this could theoretically be swapped
 * out for something that resembles ebsynth's implementation more.
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class NNFGenerator {
public:
  NNFGenerator() = default;
  virtual ~NNFGenerator() = default;

  /**
   * @brief generateNNF This is a wrapper around
   * implementationOfGenerateNNF. It currently doesn't do any error
   * checks, but I included it so that NNFGenerator's format is the same as
   * that of Downscaler, NNFUpscaler, etc.
   * @param configuration the configuration StyLit is running
   * @param pyramidLevel the pyramidLevel for which to populate forwardNNF
   * @return true if NNF generation succeeds; otherwise false
   */
  bool generateNNF(
      const Configuration &configuration,
      PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel) {
    return implementationOfGenerateNNF(configuration, pyramidLevel);
  }

protected:
  /**
   * @brief implementationOfGenerateNNF Generates a forward NNF by repeatedly
   * sampling and updating a reverse NNF. The forward NNF in the PyramidLevel
   * should be updated. This might end up needed the next-coarsest PyramidLevel
   * as an argument as well.
   * @param configuration the configuration StyLit is running
   * @param pyramidLevel the pyramidLevel for which to populate forwardNNF
   * @return true if NNF generation succeeds; otherwise false
   */
  virtual bool implementationOfGenerateNNF(
      const Configuration &configuration,
      PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel) = 0;
};

#endif // NNFGENERATOR_H

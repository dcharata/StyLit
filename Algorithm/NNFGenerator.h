#ifndef NNFGENERATOR_H
#define NNFGENERATOR_H

#include "Pyramid.h"
#include "PyramidLevel.h"

#include <QtGlobal>

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
   * @param pyramid the image pyramid
   * @param level the level of the pyramid for which the forward NNF is being
   * generated
   * @return true if NNF generation succeeds; otherwise false
   */
  bool generateNNF(const Configuration &configuration,
                   Pyramid<T, numGuideChannels, numStyleChannels> &pyramid,
                   int level, std::vector<float> &budgets) {
    // The pyramid level must be valid.
    Q_ASSERT(level >= 0 && level < int(pyramid.levels.size()));

    // The NNFs must be properly initialized.
    Q_ASSERT(pyramid.levels[level].forwardNNF.sourceDimensions.area() > 0);
    Q_ASSERT(pyramid.levels[level].forwardNNF.targetDimensions.area() > 0);
    Q_ASSERT(pyramid.levels[level].reverseNNF.sourceDimensions ==
             pyramid.levels[level].forwardNNF.sourceDimensions);
    Q_ASSERT(pyramid.levels[level].reverseNNF.sourceDimensions ==
             pyramid.levels[level].forwardNNF.targetDimensions);

    return implementationOfGenerateNNF(configuration, pyramid, level, budgets);
  }

protected:
  /**
   * @brief implementationOfGenerateNNF Generates a forward NNF by repeatedly
   * sampling and updating a reverse NNF. The forward NNF in the PyramidLevel
   * should be updated. This might end up needed the next-coarsest PyramidLevel
   * as an argument as well.
   * @param configuration the configuration StyLit is running
   * @param pyramid the image pyramid
   * @param level the level of the pyramid for which the forward NNF is being
   * generated
   * @return true if NNF generation succeeds; otherwise false
   */
  virtual bool implementationOfGenerateNNF(
      const Configuration &configuration,
      Pyramid<T, numGuideChannels, numStyleChannels> &pyramid, int level,
      std::vector<float> &budgets) = 0;
};

#endif // NNFGENERATOR_H

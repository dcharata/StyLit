#ifndef NNFGENERATORCPU_H
#define NNFGENERATORCPU_H

#include "Algorithm/NNFGenerator.h"
#include <vector>

struct Configuration;

/**
 * @brief The NNFGeneratorCPU class creates a foward NNF for
 * one iteration of Algorithm 1 in the Stylit paper
 */
template <typename T, unsigned int numGuideChannels,
          unsigned int numStyleChannels>
class NNFGeneratorCPU {
public:
  NNFGeneratorCPU() = default;
  ~NNFGeneratorCPU() = default;

private:
  /**
   * @brief implementationOfGenerateNNF Generates a forward NNF by repeatedly
   * sampling and updating a reverse NNF. The forward NNF in the PyramidLevel
   * should be updated. This might end up needed the next-coarsest PyramidLevel
   * as an argument as well.
   * @param configuration the configuration StyLit is running
   * @param pyramidLevel the pyramidLevel for which to populate forwardNNF
   * @return true if NNF generation succeeds; otherwise false
   */
   bool implementationOfGenerateNNF(const Configuration &configuration,
                                   PyramidLevel<T, numGuideChannels, numStyleChannels> &pyramidLevel) {

   }
};

#endif // NNFGENERATORCPU_H

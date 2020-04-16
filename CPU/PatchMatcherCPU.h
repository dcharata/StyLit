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

template <typename T, unsigned int numDomainChannels,
          unsigned int numCodomainChannels>
class PatchMatcherCPU : public PatchMatcher<T, numDomainChannels, numCodomainChannels> {
public:
  PatchMatcherCPU() = default;
  ~PatchMatcherCPU() = default;

private:
  /**
   * @brief implementationOfPatchMatch Runs PatchMatch to improve the specified
   * NNF. If a blacklist is specified, coordinates that map to valid coordinates
   * in blacklist should not be mapped to.
   * @param configuration the configuration StyLit is running
   * @param nnf the NNF that should be improved with PatchMatch
   * @param domainPair images representing the 'domain' of the NNF
   * @param codomainPair images representing the 'codomain' of the NNF
   * @param domainWeights weights of domain channels
   * @param codomainWeights weights of codomain channels
   * @param blacklist This optional NNF goes in the opposite direction of NNF.
   * Source coordinates that correspond to valid mappings in blacklist should
   * not be mapped to. This is used in the iterative creation of NNFs via
   * reverse NNFs in the knee point step, where parts of the forward NNF that
   * have already been mapped shouldn't be mapped again.
   * @return true if patch matching succeeds; otherwise false
   */
  bool implementationOfPatchMatch(const Configuration &configuration, NNF &nnf,
                                  const ImagePair<T, numDomainChannels> &domainPair,
                                  const ImagePair<T, numCodomainChannels> &codomainPair,
                                  const ChannelWeights<numDomainChannels> &domainWeights,
                                  const ChannelWeights<numCodomainChannels> &codomainWeights,
                                  const NNF *const blacklist = nullptr) {
    return false;
  }
};

#endif // PATCHMATCHERCPU_H

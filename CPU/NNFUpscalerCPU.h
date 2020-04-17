#ifndef NNFUPSCALERCPU_H
#define NNFUPSCALERCPU_H

#include "Configuration/Configuration.h"
#include "Algorithm/NNFUpscaler.h"

class NNF;

/**
 * @brief The NNFUpscaler class This is the interface for the
 * implementation-specific NNF upscaler, which is used to initialize PatchMatch
 * using the next-coarsest pyramid level.
 */
class NNFUpscalerCPU : public NNFUpscaler {
public:
  NNFUpscalerCPU() = default;
  virtual ~NNFUpscalerCPU() = default;

  /**
   * @brief upscaleNNF This is a wrapper around implementationOfUpscaleNNF. It
   * asserts that the NNF dimensions make sense before calling the
   * implementation-specific NNF upscaler.
   * @param configuration the configuration StyLit is running
   * @param half the NNF to upscale
   * @param full the NNF to store the result in
   * @return true if upscaling succeeds; otherwise false
   */
  bool upscaleNNFCPU(const Configuration &configuration, const NNF &half,
                  NNF &full);

protected:
  /**
   * @brief implementationOfUpscaleNNF Upscales the NNF half to twice the
   * resolution and stores the result in full.
   * @param configuration the configuration StyLit is running
   * @param half the NNF to upscale
   * @param full the NNF to store the result in
   * @return true if upscaling succeeds; otherwise false
   */
  virtual bool implementationOfUpscaleNNFCPU(const Configuration &configuration,
                                          const NNF &half, NNF &full) = 0;
};

#endif // NNFUPSCALERCPU_H

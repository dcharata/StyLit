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

protected:
  /**
   * @brief implementationOfUpscaleNNF Upscales the NNF half to twice the
   * resolution and stores the result in full.
   * @param configuration the configuration StyLit is running
   * @param half the NNF to upscale
   * @param full the NNF to store the result in
   * @return true if upscaling succeeds; otherwise false
   */
  virtual bool implementationOfUpscaleNNF(const Configuration &configuration,
                                          const NNF &half, NNF &full) override;
};

#endif // NNFUPSCALERCPU_H

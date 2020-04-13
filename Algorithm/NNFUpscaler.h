#ifndef NNFUPSCALER_H
#define NNFUPSCALER_H

#include "Configuration/Configuration.h"

class NNF;

/**
 * @brief The NNFUpscaler class This is the interface for the
 * implementation-specific NNF upscaler, which is used to initialize PatchMatch
 * using the next-coarsest pyramid level.
 */
class NNFUpscaler {
public:
  NNFUpscaler() = default;
  virtual ~NNFUpscaler() = default;

  /**
   * @brief upscaleNNF This is a wrapper around implementationOfUpscaleNNF. It
   * asserts that the NNF dimensions make sense before calling the
   * implementation-specific NNF upscaler.
   * @param configuration the configuration StyLit is running
   * @param half the NNF to upscale
   * @param full the NNF to store the result in
   * @return true if upscaling succeeds; otherwise false
   */
  bool upscaleNNF(const Configuration &configuration, const NNF &half,
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
  virtual bool implementationOfUpscaleNNF(const Configuration &configuration,
                                          const NNF &half, NNF &full) = 0;
};

#endif // NNFUPSCALER_H

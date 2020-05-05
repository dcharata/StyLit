#ifndef DOWNSCALERCUDA
#define DOWNSCALERCUDA

#include "Algorithm/Downscaler.h"
#include "Algorithm/Downscaler.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Configuration/Configuration.h"

#include "GPU/ConnectDownscalerCUDA.h"

#include <QtGlobal>
#include <stdio.h>
#include <memory>

template <typename T, unsigned int numChannels>
class DownscalerCUDA : public Downscaler<T, numChannels> {
public:
  DownscalerCUDA() = default;
  virtual ~DownscalerCUDA() = default;

protected:
  /**
   * @brief implementationOfDownscale Downscales the image full to half the
   * resolution, storing the result in the image half. Uses bilinear
   * interpolation based on EBSynth's.
   * https://github.com/jamriska/ebsynth
   * @param configuration the configuration StyLit is running
   * @param full the image to downscale from
   * @param half the image to downscale to
   * @return true if downscaling succeeds; otherwise false
   */
  bool implementationOfDownscale(const Configuration &,
                                 const Image<T, numChannels> &full,
                                 Image<T, numChannels> &half) override {
    const FeatureVector<T, numChannels> *fullData = full.data.get();
    glue();
    return false;
  }
};

#endif // DOWNSCALERCUDA

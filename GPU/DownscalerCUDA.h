#ifndef DOWNSCALERCUDA
#define DOWNSCALERCUDA

#include "Algorithm/Downscaler.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Configuration/Configuration.h"

#include <QtGlobal>
#include <memory>
#include <stdio.h>

template <typename T>
int downscaleCUDA(const T *, T *, int, int, int, int, int);

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
    // The images are converted to T pointers before being passed to CUDA.
    const T *fullData = reinterpret_cast<T *>(full.data.get());
    T *halfData = reinterpret_cast<T *>(half.data.get());
    downscaleCUDA<T>(fullData, halfData, numChannels, full.dimensions.rows,
                     full.dimensions.cols, half.dimensions.rows,
                     half.dimensions.cols);
    return false;
  }
};

#endif // DOWNSCALERCUDA

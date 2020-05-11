#include "NNFUpscaler.h"

#include "NNF.h"

#include <QtGlobal>

bool NNFUpscaler::upscaleNNF(const Configuration &configuration, const NNF &half, NNF &full) {
  assert(half.sourceDimensions.halfTheSizeOf(full.sourceDimensions) &&
         half.targetDimensions.halfTheSizeOf(full.targetDimensions));
  return implementationOfUpscaleNNF(configuration, half, full);
}

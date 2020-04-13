#include "NNFUpscaler.h"

#include "NNF.h"

#include <QtGlobal>

bool NNFUpscaler::upscaleNNF(const Configuration &configuration,
                             const NNF &half, NNF &full) {
  assert(half.sourceDimensions.halfTheSizeOf(full.sourceDimensions) &&
         half.targetDimensions.halfTheSizeOf(full.targetDimensions));
  return implementationOfUpscaleNNF(configuration, half, full);
}

bool NNFUpscaler::implementationOfUpscaleNNF(const Configuration &configuration,
                                const NNF &half, NNF &full) {


    for (int i = 0; i < full.targetDimensions.rows; i++) {
        for (int j = 0; j < full.targetDimensions.cols; j++) {
            ImageCoordinates temp = half.getMapping( {qBound(0, i/2, half.targetDimensions.rows - 1),
                                                      qBound(0, j/2, half.targetDimensions.cols - 1)} ) * 2;

            // 5 is the patch size
            full.setMapping({i,j}, {qBound(5, temp.row, full.sourceDimensions.rows - 5 - 1),
                            qBound(5, temp.col, full.sourceDimensions.cols - 5 - 1)});
        }
    }

    return 1;
}

#include "NNFUpscalerCPU.h"

#include "Algorithm/NNF.h"

#include <QtGlobal>


bool NNFUpscalerCPU::implementationOfUpscaleNNF(const Configuration &configuration,
                                const NNF &half, NNF &full) {


    for (int i = 0; i < full.targetDimensions.rows; i++) {
        for (int j = 0; j < full.targetDimensions.cols; j++) {

            const int _i = qBound(0, i/2, half.targetDimensions.rows - 1);
            const int _j = qBound(0, j/2, half.targetDimensions.cols - 1);
            const ImageCoordinates _coord{_i, _j};
            const ImageCoordinates offset{i % 2, j % 2};
            ImageCoordinates temp = half.getMapping(_coord) * 2 + offset;

            // 5 is the patch size
            const int patch_size = 5;
            const ImageCoordinates coord{i, j};
            const int vi = qBound(patch_size, temp.row, full.sourceDimensions.rows - patch_size - 1);
            const int vj = qBound(patch_size, temp.col, full.sourceDimensions.cols - patch_size - 1);
            const ImageCoordinates value{vi, vj};
            full.setMapping(coord, value);
        }
    }

    return 1;
}

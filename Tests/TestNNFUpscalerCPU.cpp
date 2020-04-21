#include <vector>
#include "TestNNFUpscalerCPU.h"
#include "CPU/NNFUpscalerCPU.h"
#include "Algorithm/NNF.h"
#include "Configuration/Configuration.h"

// Pointless test
bool TestNNFUpscalerCPU::run() {

    NNFUpscalerCPU nnfUpscalerCPU;
    Configuration configuration;

    std::vector<int> patch_sizes = {0, 20, 50, 100};

    ImageDimensions sourceDimensions{200,150};
    ImageDimensions targetDimensions = sourceDimensions;

    NNF half{sourceDimensions, targetDimensions};
    NNF full{sourceDimensions * 2, targetDimensions * 2};
    NNF verify{sourceDimensions * 2, targetDimensions * 2};

    // Initialize half NNF with indentity map
    for(int i = 0; i < half.targetDimensions.rows; i++) {
        for (int j = 0; j < half.targetDimensions.cols; j++) {
            half.setMapping({i,j},{i,j});
        }
    }

    // Iterate through different patch sizes
    for(int ps : patch_sizes)
    {
        configuration.patchSize = ps;

        // Initialize a verification NNF
        for(int i = 0; i < verify.targetDimensions.rows; i++) {
            for (int j = 0; j < verify.targetDimensions.cols; j++) {
                int vi = (i < configuration.patchSize) ? configuration.patchSize : i;
                vi = (vi > (verify.sourceDimensions.rows - configuration.patchSize - 1)) ?
                          (verify.sourceDimensions.rows - configuration.patchSize - 1) : vi;
                int vj = (j < configuration.patchSize) ? configuration.patchSize : j;
                vj = (vj > (verify.sourceDimensions.cols - configuration.patchSize - 1)) ?
                          (verify.sourceDimensions.cols - configuration.patchSize - 1) : vj;
                verify.setMapping({i,j},{vi,vj});
            }
        }

        TEST_ASSERT(nnfUpscalerCPU.upscaleNNF(configuration, half, full));

        for(int i = 0; i < full.targetDimensions.rows; i++) {
            for (int j = 0; j < full.targetDimensions.cols; j++) {
                TEST_ASSERT(full.getMapping({i,j}) == verify.getMapping({i,j}));
            }
        }
    }

    return true;
}

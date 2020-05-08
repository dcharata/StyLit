#include "TestPatchMatchCUDA.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Algorithm/Pyramid.h"
#include "CPU/NNFApplicatorCPU.h"
#include "GPU/PatchMatcherCUDA.h"
#include "Utilities/FloatTools.h"
#include "Utilities/ImageFormat.h"
#include "Utilities/ImageIO.h"
#include <QFile>
#include <QImage>
#include <iostream>

bool TestPatchMatchCUDA::run() {

  // run patchmatch and make sure that the error gets lower with each iteration
  // run patchmatch where source and target are similar, output image and see if
  // image looks like target

  Configuration configuration;
  configuration.patchSize = 5;

  {
    const QString path1("./Examples/brown1.png");
    const QString path2("./Examples/brown2.png");
    QImage sourceImage(path1);

    const ImageDimensions sourceDims{sourceImage.height(), sourceImage.width()};
    const QImage targetImage(path2);
    const ImageDimensions targetDims{targetImage.height(), targetImage.width()};
    Pyramid<float, 3, 3> pyramid;
    const ChannelWeights<3> guideWeights(1.0, 1.0, 1.0);
    const ChannelWeights<3> styleWeights(1.0, 1.0, 1.0);
    pyramid.guideWeights = guideWeights;
    pyramid.styleWeights = styleWeights;
    pyramid.levels.push_back(PyramidLevel<float, 3, 3>(sourceDims, targetDims));

    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].guide.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].style.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].guide.target, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].style.target, ImageFormat::RGB, 0));
    PatchMatcherCUDA<float, 3, 3> patchMatcher;
    ErrorCalculatorCPU<float, 3, 3> errorCalc;
    patchMatcher.randomlyInitializeNNF(pyramid.levels[0].forwardNNF);
    float totalError = 0;
    for (int col = 0; col < pyramid.levels[0].forwardNNF.sourceDimensions.cols; col++) {
      for (int row = 0; row < pyramid.levels[0].forwardNNF.sourceDimensions.rows; row++) {
        float error = 0;
        ImageCoordinates coords = {row, col};
        errorCalc.calculateError(configuration, pyramid.levels[0],
                                 pyramid.levels[0].forwardNNF.getMapping(coords), coords,
                                 guideWeights, styleWeights, error);
        totalError += error;
      }
    }
    std::cout << "Error: " << totalError << std::endl;
    for (int i = 0; i < 2; i++) {
      patchMatcher.patchMatch(configuration, pyramid.levels[0].forwardNNF, pyramid, 2, 0, false,
                              false);
      float totalError = 0;
      for (int col = 0; col < pyramid.levels[0].forwardNNF.sourceDimensions.cols; col++) {
        for (int row = 0; row < pyramid.levels[0].forwardNNF.sourceDimensions.rows; row++) {
          // Runs a sanity check to make sure the data coming back from the GPU isn't complete
          // garbage.
          ImageCoordinates c = pyramid.levels[0].forwardNNF.getMapping({row, col});
          TEST_ASSERT(c.row >= 0 && c.col >= 0 &&
                      c.row < pyramid.levels[0].guide.source.dimensions.rows &&
                      c.col < pyramid.levels[0].guide.source.dimensions.cols);

          float error = 0;
          ImageCoordinates coords = {row, col};
          errorCalc.calculateError(configuration, pyramid.levels[0],
                                   pyramid.levels[0].forwardNNF.getMapping(coords), coords,
                                   guideWeights, styleWeights, error);
          totalError += error;
        }
      }
      std::cout << "Error: " << totalError << std::endl;
    }

    NNFApplicatorCPU<float, 3, 3> imageMaker;
    imageMaker.applyNNF(configuration, pyramid.levels[0]);
    ImageIO::writeImage("./Examples/patchMatchTest1Output.png", pyramid.levels[0].style.target,
                        ImageFormat::RGB, 0);
  }
  return true;
}

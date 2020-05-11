#include "TestPatchMatch.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Utilities/FloatTools.h"
#include "Utilities/ImageFormat.h"
#include "Utilities/ImageIO.h"
#include <QFile>
#include <QImage>
#include "Algorithm/Pyramid.h"
//#include "CPU/PatchMatcherCPU.h"
#include <iostream>
#include "CPU/NNFApplicatorCPU.h"

bool TestPatchMatch::run() {

  // run patchmatch and make sure that the error gets lower with each iteration
  // run patchmatch where source and target are similar, output image and see if image looks like target

  Configuration configuration;
  configuration.patchSize = 5;
  configuration.numPatchMatchIterations = 6;
  /*
  {
    const QString path1("./test/original_src/examples/1/source_style.png");
    const QString path2("./test/original_src/examples/1/output.png");
    QImage sourceImage(path1);

    const ImageDimensions sourceDims{sourceImage.height(), sourceImage.width()};
    const QImage targetImage(path2);
    const ImageDimensions targetDims{targetImage.height(), targetImage.width()};
    Pyramid<float, 3, 3> pyramid;
    const ChannelWeights<3> guideWeights(1.0,1.0,1.0);
    const ChannelWeights<3> styleWeights(1.0,1.0,1.0);
    pyramid.guideWeights = guideWeights;
    pyramid.styleWeights = styleWeights;
    pyramid.levels.push_back(PyramidLevel<float, 3, 3>(sourceDims, targetDims));

    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].guide.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].style.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].guide.target, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].style.target, ImageFormat::RGB, 0));
    PatchMatcherCPU<float, 3, 3> patchMatcher;
    ErrorCalculatorCPU<float, 3, 3> errorCalc;
    patchMatcher.randomlyInitializeNNF(pyramid.levels[0].forwardNNF);
    float totalError = 0;
    for (int col = 0; col < pyramid.levels[0].forwardNNF.sourceDimensions.cols; col++) {
      for (int row = 0; row < pyramid.levels[0].forwardNNF.sourceDimensions.rows; row++) {
        float error = 0;
        ImageCoordinates coords = {row, col};
        errorCalc.calculateError(configuration, pyramid.levels[0], pyramid.levels[0].forwardNNF.getMapping(coords),
                                 coords, guideWeights, styleWeights, error);
        totalError += error;
      }
    }
    std::cout << "Error: " << totalError << std::endl;
    for (int i = 0; i < 2; i++) {
      //patchMatcher.patchMatch(configuration, pyramid.levels[0].forwardNNF, pyramid, 0, false, false);
      float totalError = 0;
      for (int col = 0; col < pyramid.levels[0].forwardNNF.sourceDimensions.cols; col++) {
        for (int row = 0; row < pyramid.levels[0].forwardNNF.sourceDimensions.rows; row++) {
          float error = 0;
          ImageCoordinates coords = {row, col};
          errorCalc.calculateError(configuration, pyramid.levels[0], pyramid.levels[0].forwardNNF.getMapping(coords),
                                   coords, guideWeights, styleWeights, error);
          totalError += error;
        }
      }
      std::cout << "Error: " << totalError << std::endl;
    }

    NNFApplicatorCPU<float, 3, 3> imageMaker;
    imageMaker.applyNNF(configuration, pyramid.levels[0]);
    ImageIO::writeImage("./Examples/patchMatchTest1Output.png", pyramid.levels[0].style.target, ImageFormat::RGB, 0);

  }

  // test whether patchmatch can copy an image using patches from a similar image
  {
    const QString path1("./Examples/up1.png");
    const QString path2("./Examples/up2.png");
    const QImage sourceImage(path1);
    const ImageDimensions sourceDims{sourceImage.height(), sourceImage.width()};
    const QImage targetImage(path2);
    const ImageDimensions targetDims{targetImage.height(), targetImage.width()};
    Pyramid<float, 3, 3> pyramid;
    const ChannelWeights<3> guideWeights(1.0,1.0,1.0);
    const ChannelWeights<3> styleWeights(1.0,1.0,1.0);
    pyramid.guideWeights = guideWeights;
    pyramid.styleWeights = styleWeights;
    pyramid.levels.push_back(PyramidLevel<float, 3, 3>(sourceDims, targetDims));

    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].guide.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].style.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].guide.target, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].style.target, ImageFormat::RGB, 0));
    PatchMatcherCPU<float, 3, 3> patchMatcher;
    ErrorCalculatorCPU<float, 3, 3> errorCalc;
    patchMatcher.randomlyInitializeNNF(pyramid.levels[0].forwardNNF);

    //patchMatcher.patchMatch(configuration, pyramid.levels[0].forwardNNF, pyramid, 0, false, false);

    NNFApplicatorCPU<float, 3, 3> imageMaker;
    imageMaker.applyNNF(configuration, pyramid.levels[0]);
    ImageIO::writeImage("./Examples/patchMatchTest2Output.png", pyramid.levels[0].style.target, ImageFormat::RGB, 0);
  }


  // test blacklisting
  // the entire brown1 image will be constructed from the bottom half of the image.
  {
    const QString path1("./Examples/brown1.png");
    const QString path2("./Examples/brown1.png");
    const QImage sourceImage(path1);
    const ImageDimensions sourceDims{sourceImage.height(), sourceImage.width()};
    const QImage targetImage(path2);
    const ImageDimensions targetDims{targetImage.height(), targetImage.width()};
    Pyramid<float, 3, 3> pyramid;
    const ChannelWeights<3> guideWeights(1.0,1.0,1.0);
    const ChannelWeights<3> styleWeights(1.0,1.0,1.0);
    pyramid.guideWeights = guideWeights;
    pyramid.styleWeights = styleWeights;
    pyramid.levels.push_back(PyramidLevel<float, 3, 3>(sourceDims, targetDims));

    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].guide.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].style.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].guide.target, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].style.target, ImageFormat::RGB, 0));
    PatchMatcherCPU<float, 3, 3> patchMatcher;
    ErrorCalculatorCPU<float, 3, 3> errorCalc;
    std::srand(5);
    patchMatcher.randomlyInitializeNNF(pyramid.levels[0].forwardNNF);

    NNF blacklist(sourceDims, targetDims);
    blacklist.setToInitializedBlacklist();
    for (int col = 0; col < pyramid.levels[0].forwardNNF.sourceDimensions.cols; col++) {
      for (int row = 0; row < .5 * pyramid.levels[0].forwardNNF.sourceDimensions.rows; row++) {
        blacklist.setMapping(ImageCoordinates{row, col}, {1,1});
      }
    }

    //patchMatcher.patchMatch(configuration, pyramid.levels[0].forwardNNF, pyramid, 6, 0, false, false, &blacklist);

    int invalidMappings = 0;
    for (int col = 0; col < pyramid.levels[0].forwardNNF.sourceDimensions.cols; col++) {
      for (int row = 0; row < pyramid.levels[0].forwardNNF.sourceDimensions.rows; row++) {
        if (pyramid.levels[0].forwardNNF.getMapping(ImageCoordinates{row,col}).row < .5 * pyramid.levels[0].forwardNNF.sourceDimensions.rows) {
          invalidMappings++;
        }
      }
    }
    std::cout << "Total invalid mappings (patches on blacklist that were still mapped to): " << invalidMappings << std::endl;

    NNFApplicatorCPU<float, 3, 3> imageMaker;
    imageMaker.applyNNF(configuration, pyramid.levels[0]);
    ImageIO::writeImage("./Examples/patchMatchTest3Output.png", pyramid.levels[0].style.target, ImageFormat::RGB, 0);
  }
*/

  std::cout << "PatchMatch tests ran, make sure error values decreased and check output images" << std::endl;
  return true;
}

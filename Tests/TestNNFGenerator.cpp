#include "TestNNFGenerator.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Utilities/FloatTools.h"
#include "Utilities/ImageFormat.h"
#include "Utilities/ImageIO.h"
#include <QFile>
#include <QImage>
#include "Algorithm/Pyramid.h"
#include "CPU/PatchMatcherCPU.h"
#include <iostream>
#include "CPU/NNFApplicatorCPU.h"
#include "CPU/NNFGeneratorCPU.h"

bool TestNNFGenerator::run() {
  QString path1("./Examples/brown2.png");
  QString path2("./Examples/brown1.png");

  // run patchmatch and make sure that the error gets lower with each iteration
  // run patchmatch where source and target are similar, output image and see if image looks like target
  {
    QImage sourceImage(path1);
    ImageDimensions sourceDims{sourceImage.height(), sourceImage.width()};
    QImage targetImage(path2);
    ImageDimensions targetDims{targetImage.height(), targetImage.width()};
    Pyramid<float, 3, 3> pyramid;
    ChannelWeights<3> guideWeights(1.0,1.0,1.0);
    ChannelWeights<3> styleWeights(1.0,1.0,1.0);
    pyramid.guideWeights = guideWeights;
    pyramid.styleWeights = styleWeights;
    pyramid.levels.push_back(PyramidLevel<float, 3, 3>(sourceDims, targetDims));

    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].guide.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path1, pyramid.levels[0].style.source, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].guide.target, ImageFormat::RGB, 0));
    TEST_ASSERT(ImageIO::readImage<3>(path2, pyramid.levels[0].style.target, ImageFormat::RGB, 0));

    PatchMatcherCPU<float, 3, 3> patchMatcher;
    patchMatcher.randomlyInitializeNNF(pyramid.levels[0].reverseNNF);






    NNFApplicatorCPU<float, 3, 3> imageMaker;
    imageMaker.applyNNF(Configuration(), pyramid.levels[0]);
    ImageIO::writeImage("./Examples/patchMatchOutput.png", pyramid.levels[0].style.target, ImageFormat::RGB, 0);

  }



  return true;
}

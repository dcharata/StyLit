#include "TestNNFGenerator.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Utilities/FloatTools.h"
#include "Utilities/ImageFormat.h"
#include "Utilities/ImageIO.h"
#include <QFile>
#include <QImage>
#include "Algorithm/Pyramid.h"
#include <iostream>
#include "CPU/NNFApplicatorCPU.h"

bool TestNNFGenerator::run() {

  // generate forward NNF from reverse NNFs using the NNFGenerator
  {
    Configuration configuration;
    configuration.patchSize = 5;
    configuration.numPatchMatchIterations = 6;

    {
      const QString path1("./Examples/brown1.png");
      const QString path2("./Examples/brown2.png");
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
      //NNFGeneratorCPU<float, 3, 3> generator;
      //generator.generateNNF(configuration, pyramid, 0);
      NNFApplicatorCPU<float, 3, 3> imageMaker;
      imageMaker.applyNNF(configuration, pyramid.levels[0]);
      ImageIO::writeImage("./Examples/NNFGeneratorTest1Output.png", pyramid.levels[0].style.target, ImageFormat::RGB, 0);

    }
  }

  return true;
}

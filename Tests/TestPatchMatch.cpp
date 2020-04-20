#include "TestPatchMatch.h"
#include "Algorithm/Image.h"
#include "Algorithm/ImageDimensions.h"
#include "Utilities/FloatTools.h"
#include "Utilities/ImageFormat.h"
#include "Utilities/ImageIO.h"
#include <QFile>
#include "Algorithm/Pyramid.h"

bool TestPatchMatch::run() {
  QString path("./Examples/brown.png");

  // run patchmatch and make sure that the error gets lower with each iteration
  {
    Pyramid<float, 3, 3> pyramid;
    ImageDimensions dims{586, 1148};
    Image<float, 3> A(dims);
    ImageIO::readImage<3>(path, A, ImageFormat::RGB, 0);
    Image<float, 3> A_prime(dims);
    ImageIO::readImage<3>(path, A_prime, ImageFormat::RGB, 0);
    Image<float, 3> B(dims);
    ImageIO::readImage<3>(path, B, ImageFormat::RGB, 0);
    Image<float, 3> B_prime(dims);
    ImageIO::readImage<3>(path, B_prime, ImageFormat::RGB, 0);
    ImagePair<float, 3> pair(dims);
    PyramidLevel<float, 3, 3> level(dims, dims);
  }

  return true;
}

#include "TestImageResize.h"

//#include "Algorithm/Image.h"
//#include "Algorithm/ImageDimensions.h"
//#include "Utilities/ImageFormat.h"
//#include "Utilities/ImageIO.h"

//#include "Configuration/Configuration.h"
//#include "Configuration/ConfigurationParser.h"

//#include "CPU/DownscalerCPU.h"

bool TestImageResize::run() {

//    QString path("./Examples/baboon.png");
    // Gets the image dimensions.
//    ImageDimensions dimensionsFull;
//    ImageDimensions dimensionsHalf;
//    TEST_ASSERT(ImageIO::getImageDimensions(path, dimensionsFull));
//    TEST_ASSERT(dimensionsFull.rows == 512);
//    TEST_ASSERT(dimensionsFull.cols == 512);

//    dimensionsHalf.rows = dimensionsFull.rows / 2;
//    dimensionsHalf.cols = dimensionsFull.cols / 2;

//    // Read RGB image.
//    Image<float, 3> fullSize(dimensionsFull);
//    Image<float, 3> halfSize(dimensionsHalf);
//    TEST_ASSERT(ImageIO::readImage<3>(path,
//                                      fullSize, ImageFormat::RGB, 0));

//    // Reads the configuration.
//    QString configurationPath("./Examples/example-configuratation.json");
//    Configuration configuration;
//    ConfigurationParser configurationParser(configurationPath);
//    if (!configurationParser.parse(configuration)) {
//      return false;
//    }

    // downscale and write

    return true;
}

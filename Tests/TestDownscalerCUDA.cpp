#include "TestDownscalerCUDA.h"

#include "GPU/DownscalerCUDA.h"
#include "TestDownscaler.h"
#include "TestDownscalerWithImage.h"

#include <iostream>

bool TestDownscalerCUDA::run() {
  // TestDownscaler<DownscalerCUDA> basicTest;
  TestDownscalerWithImage<DownscalerCUDA> imageTest;
  // return basicTest.run() && imageTest.run();
  return imageTest.run();
}

#include "TestDownscalerCUDA.h"

#include "GPU/DownscalerCUDA.h"
#include "TestDownscaler.h"
#include "TestDownscalerWithImage.h"

#include <iostream>

bool TestDownscalerCUDA::run() {
  TestDownscaler<DownscalerCUDA> basicTest;
  TestDownscalerWithImage<DownscalerCUDA> imageTest;

  std::cout << "HI" << std::endl;

  return basicTest.run() && imageTest.run();
}

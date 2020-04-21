#include "TestDownscalerCPU.h"

#include "CPU/DownscalerCPU.h"
#include "TestDownscaler.h"
#include "TestDownscalerWithImage.h"

bool TestDownscalerCPU::run() {
  TestDownscaler<DownscalerCPU> basicTest;
  TestDownscalerWithImage<DownscalerCPU> imageTest;
  return basicTest.run() && imageTest.run();
}

#include "StyLitCUDA.h"

namespace StyLitCUDA {
  void runCoordinator_float(InterfaceInput<float> &input);
  void runEBSynthCoordinator_float(InterfaceInput<float> &input);
}

unsigned int StyLitCUDA_sanityCheckStyLitCUDA() { return 0xDEADBEEF; }

int StyLitCUDA_runStyLitCUDA_float(StyLitCUDA::InterfaceInput<float> &input) {
  //StyLitCUDA::runCoordinator_float(input);
  StyLitCUDA::runEBSynthCoordinator_float(input);
  return 503;
}

#ifndef TESTDOWNSCALERCUDA_H
#define TESTDOWNSCALERCUDA_H

#include "UnitTest.h"

class TestDownscalerCUDA : public UnitTest {
public:
  TestDownscalerCUDA() = default;
  bool run() override;
};

#endif // TESTDOWNSCALERCUDA_H

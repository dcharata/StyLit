#ifndef TESTPATCHMATCHCUDA_H
#define TESTPATCHMATCHCUDA_H

#include "UnitTest.h"

class TestPatchMatchCUDA : public UnitTest {
public:
  TestPatchMatchCUDA() = default;
  bool run() override;
};

#endif // TESTPATCHMATCHCUDA_H

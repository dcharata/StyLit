#ifndef TESTCUDA_H
#define TESTCUDA_H

#include "UnitTest.h"

class TestCuda : public UnitTest
{
public:
  TestCuda() = default;
  bool run() override;
};

#endif // TESTCUDA_H

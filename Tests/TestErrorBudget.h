#ifndef TESTERRORBUDGET_H
#define TESTERRORBUDGET_H

#include "UnitTest.h"

class TestErrorBudget : public UnitTest {
public:
  TestErrorBudget() = default;
  bool run() override;
};

#endif // TESTERRORBUDGET_H

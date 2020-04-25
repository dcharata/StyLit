#ifndef ERRORBUDGETCALCULATORCPU_H
#define ERRORBUDGETCALCULATORCPU_H

#include "Algorithm/ErrorBudgetCalculator.h"
#include "Algorithm/NNF.h"

/**
 * @brief Implementation of Kneepoint finder class for CPU
 */
class ErrorBudgetCalculatorCPU : public ErrorBudgetCalculator {
public:
  ErrorBudgetCalculatorCPU() = default;
  virtual ~ErrorBudgetCalculatorCPU() = default;

private:
  /**
   * @brief implementationOfCalculateErrorBudget Finds the knee point for the
   * given errors, then uses it to determine the error budget. Returns the error
   * budget through an out argument.
   * @param configuration the configuration StyLit is running
   * @param error the 2D array of errors used to calculate the knee point
   * @param errorBudget the resulting error budget
   * @return true if error budget calculation succeeds; otherwise false
   */
  bool implementationOfCalculateErrorBudget(const Configuration &configuration,
                                            std::vector<std::pair<int, float>> &vecerror,
                                            const NNFError &error,
                                            const float totalError,
                                            float &errorBudget,
                                            const NNF *const blacklist = nullptr) override;
};

#endif // ERRORBUDGETCALCULATORCPU_H

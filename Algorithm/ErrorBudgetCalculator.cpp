#include "ErrorBudgetCalculator.h"

bool ErrorBudgetCalculator::calculateErrorBudget(
    const Configuration &configuration, const NNFError &error,
    float errorBudget) {
  return implementationOfCalculateErrorBudget(configuration, error,
                                              errorBudget);
}

#ifndef ERRORBUDGETCALCULATOR_H
#define ERRORBUDGETCALCULATOR_H

#include <vector>

struct Configuration;
struct NNFError;
struct NNF;


/**
 * @brief The KneePointFinder class This fits a hyperbolic function to a sorted
 * array of the errors and uses it to calculate an error budget for the patches
 * in the reverse NNF that are used to construct the forward NNF.
 */
class ErrorBudgetCalculator {
public:
  ErrorBudgetCalculator() = default;
  virtual ~ErrorBudgetCalculator() = default;

  /**
   * @brief calculateErrorBudget This is a wrapper around
   * implementationOfCalculateErrorBudget. It currently doesn't do any error
   * checks, but I included it so that KneePointFinder's format is the same as
   * that of Downscaler, NNFUpscaler, etc.
   * @param configuration the configuration StyLit is running
   * @param error the 2D array of errors used to calculate the knee point
   * @param errorBudget the resulting error budget
   * @return true if error budget calculation succeeds; otherwise false
   */
  bool calculateErrorBudget(const Configuration &configuration,
                            std::vector<std::pair<int, float>> &vecerror,
                            const NNFError &error,
                            const float totalError,
                            float &errorBudget,
                            const NNF *const blacklist = nullptr) {
    return implementationOfCalculateErrorBudget(configuration, vecerror, error, totalError, errorBudget, blacklist);
  }

protected:
  /**
   * @brief implementationOfCalculateErrorBudget Finds the knee point for the
   * given errors, then uses it to determine the error budget. Returns the error
   * budget through an out argument.
   * @param configuration the configuration StyLit is running
   * @param error the 2D array of errors used to calculate the knee point
   * @param errorBudget the resulting error budget
   * @return true if error budget calculation succeeds; otherwise false
   */
  virtual bool implementationOfCalculateErrorBudget(const Configuration &configuration,
                                            std::vector<std::pair<int, float>> &vecerror,
                                            const NNFError &error,
                                            const float totalError,
                                            float &errorBudget,
                                            const NNF *const blacklist = nullptr) = 0;
};

#endif // ERRORBUDGETCALCULATOR_H

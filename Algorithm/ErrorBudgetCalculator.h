#ifndef ERRORBUDGETCALCULATOR_H
#define ERRORBUDGETCALCULATOR_H

//#define DLIB_NO_GUI_SUPPORT
#include <dlib/optimization.h>

using namespace dlib;

struct Configuration;
struct NNFError;

typedef double input_vector;
typedef matrix<double,2,1> parameter_vector;

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
                            const NNFError &error, float errorBudget);

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
  bool
  implementationOfCalculateErrorBudget(const Configuration &configuration,
                                       const NNFError &error,
                                       float errorBudget);

};

#endif // ERRORBUDGETCALCULATOR_H

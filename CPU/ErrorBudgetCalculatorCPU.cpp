#include "ErrorBudgetCalculatorCPU.h"

#include <Eigen/Eigen>
#include <iostream>
#include <unsupported/Eigen/NonLinearOptimization>
#include <vector>

#include "Algorithm/NNFError.h"
#include "Configuration/Configuration.h"
//#include "Utilities/parasort.h"
//#include <parallel/algorithm>

#include <fstream>
#include <limits>

// ----------------------------------------------------------------------------------------
// hyperbolic function
struct LMFunctor {
  // 'm' pairs of (x, f(x))
  Eigen::MatrixXd measuredValues;

  // Compute 'm' errors, one for each data point, for the given parameter values
  // in 'x'
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const {
    // 'x' has dimensions n x 1
    // It contains the current estimates for the parameters.

    // 'fvec' has dimensions m x 1
    // It will contain the error for each data point.

    const double aParam = x(0);
    const double bParam = x(1);

    for (int i = 0; i < values(); i++) {
      double xValue = measuredValues(i, 0);
      double yValue = measuredValues(i, 1);

      fvec(i, 0) = yValue - powf(aParam - bParam * xValue, -1);
    }
    return 0;
  }

  // Compute the jacobian of the errors
  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const {
    // 'x' has dimensions n x 1
    // It contains the current estimates for the parameters.

    // 'fjac' has dimensions m x n
    // It will contain the jacobian of the errors, calculated numerically in
    // this case.

    const double epsilon = 1e-6f;

    // numerical Jacobian
    for (int in = 0; in < x.size(); in++) {
      Eigen::VectorXd xPlus(x);
      xPlus(in) += epsilon;
      Eigen::VectorXd xMinus(x);
      xMinus(in) -= epsilon;

      Eigen::VectorXd fvecPlus(values());
      operator()(xPlus, fvecPlus);

      Eigen::VectorXd fvecMinus(values());
      operator()(xMinus, fvecMinus);

      Eigen::VectorXd fvecDiff(values());
      fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

      fjac.block(0, in, values(), 1) = fvecDiff;
    }

    // analytical Jacobian
    //        double aParam = x(0);
    //        double bParam = x(1);
    //        for (int im=0; im<values(); im++) {
    //            double temp = -1.0 *powf(aParam - bParam * measuredValues(im,
    //            0), -2); fjac(im, 0) = temp; // da fjac(im, 1) = -1.0 *
    //            measuredValues(im, 0) * temp; // db
    //        }

    return 0;
  }

  // Number of data points, i.e. values.
  int m;

  // Returns 'm', the number of values.
  int values() const { return m; }

  // The number of parameters, i.e. inputs.
  int n;

  // Returns 'n', the number of inputs.
  int inputs() const { return n; }
};

bool comparator(const std::pair<int, float> lhs,
                const std::pair<int, float> rhs) {
  return lhs.second < rhs.second;
}

// ----------------------------------------------------------------------------------------

bool ErrorBudgetCalculatorCPU::implementationOfCalculateErrorBudget(
    const Configuration &config,
    const std::vector<std::pair<float, ImageCoordinates>> &vecerror,
    const NNFError &nnferror, const float totalError, float &errorBudget,
    const NNF *const blacklist) {

  // read from the error image
  const int height = nnferror.error.dimensions.rows;
  const int width = nnferror.error.dimensions.cols;
  const int num_pixels = height * width;

  float meanError = totalError / vecerror.size();

  std::cout << "meanError " << meanError << std::endl;

  std::cout << "vecErrorSize " << vecerror.size() << std::endl;

  std::cout << "totalError " << totalError << std::endl;

  // writes the errors to CSV for graphing
  // std::ofstream
  // outFile("/Users/davidcharatan/Documents/StyLitBin/errors.csv");
  // for (const auto &e : vecerror)
  // outFile << e.second << ", ";
  // outFile << std::endl;

  int NUM_SAMPLES = 50;

  Eigen::MatrixXd measuredValues(NUM_SAMPLES, 2); // pairs of (x, f(x))
  double x_scale = 1.f / double(height * width);
  int idx = 0;
  int validSamples = 0;
  for (unsigned int i = 0; i < vecerror.size(); i++) {
    if (i % ((height * width) / (NUM_SAMPLES - 1)) == 0 &&
        (vecerror[i].first < 100.0f)) {
      // normalize the x axis
      measuredValues(idx, 0) = float(i) * x_scale;
      // measuredValues(i, 0) = float(i) / float(vecerror.size());

      // normalize the y axis
      measuredValues(idx, 1) = (double)vecerror[i].first / double(meanError);
      idx++;
      validSamples++;
    }
    // std::cout << vecerror[i].first << std::endl;
  }

  int n = 2; // number of parameters

  // 'params' is vector of length 'n' containing the initial values for the
  // parameters.
  Eigen::VectorXd params(n);
  // initialization
  params(0) = 1.f; // a
  params(1) = 1.f; // b

  // Run the LM optimization
  // Create a LevenbergMarquardt object and pass it the functor.
  LMFunctor functor;
  functor.measuredValues = measuredValues;
  functor.m = validSamples; // num_pixels;
  functor.n = n;

  Eigen::LevenbergMarquardt<LMFunctor, double> lm(functor);
  std::cout << "minimizing" << std::endl;
  lm.minimize(params); // TODO: Use the status for something.
  std::cout << "minimizing" << std::endl;

  // calculate the knee point
  double a = params(0);
  double b = params(1);

  std::cout << "Value of a: " << a << std::endl;
  std::cout << "Value of b: " << b << std::endl;

  double kneepoint;
  if (b < 0) {
    kneepoint = (sqrtf(1.f / b) + a / b);
    std::cout << "the b term in the function-fitting step is negative, which "
                 "shouldn't happen"
              << std::endl;
  } else {
    kneepoint = (-sqrtf(1.f / b) +
                 a / b); // this is the case that should normally happen
  }

  std::cout << "kneepoint " << kneepoint << std::endl;

  // get the kneepoint index
  // we need to multply by the number of pixels to undo the normalization
  int kneepointIndex = std::max<int>(
      0, std::min<int>(int(kneepoint * (validSamples)), validSamples - 1));
  std::cout << "Kneepoint index: " << kneepointIndex << std::endl;

  // we need to multiply by the mean error to undo the normalization
  errorBudget = measuredValues(kneepointIndex, 1) * meanError;

  return true;
}

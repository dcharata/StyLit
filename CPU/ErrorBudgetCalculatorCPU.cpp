#include "ErrorBudgetCalculatorCPU.h"

#include <Eigen/Eigen>
#include <iostream>
#include <unsupported/Eigen/NonLinearOptimization>
#include <vector>

#include "Algorithm/NNFError.h"
#include "Configuration/Configuration.h"

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
    const Configuration &config, std::vector<std::pair<int, float>> &vecerror,
    const NNFError &nnferror, const float totalError, float &errorBudget,
    const NNF *const blacklist) {
  // we may not need configuration?

  // read from the error image
  const int height = nnferror.error.dimensions.rows;
  const int width = nnferror.error.dimensions.cols;
  const int num_pixels = height * width;
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      ImageCoordinates codomainPatch =
          nnferror.nnf.getMapping(ImageCoordinates{row, col});
      bool validMapping =
          (blacklist == nullptr) || (blacklist->getMapping(codomainPatch) ==
                                     ImageCoordinates::FREE_PATCH);
      if (validMapping) {
        vecerror.push_back(std::make_pair(
            row * width + col, nnferror.error.getConstPixel(row, col)[0]));
      }
    }
  }

  // sort the error vector
  sort(vecerror.begin(), vecerror.end(), &comparator);

  // float maxError = vecerror[vecerror.size() - 1].second;
  float meanError = totalError / vecerror.size();
  float patchSizeSquared = config.patchSize * config.patchSize;

  std::cout << "Mean error" << meanError << std::endl;

  std::cout << "Max error: " << vecerror[vecerror.size() - 1].second
            << std::endl;

  // convert to eigen matrix
  // ref:
  // https://medium.com/@sarvagya.vaish/levenberg-marquardt-optimization-part-2-5a71f7db27a0
  Eigen::MatrixXd measuredValues(vecerror.size(), 2); // pairs of (x, f(x))
  double x_scale = 1.f / double(height * width);
  for (unsigned int i = 0; i < vecerror.size(); i++) {
    // normalize the x axis
    measuredValues(i, 0) = float(i) * x_scale;

    // divide by total error to normalize the y axis
    measuredValues(i, 1) = (double)vecerror[i].second / patchSizeSquared;
    // measuredValues(i, 1) = (double)vecerror[i].second / double(meanError);
    // measuredValues(i, 1) = (double)vecerror[i].second / double(totalError);
  }

  // fit the hyperbolic function
  // use the Levenberg-Marquardt method to determine the parameters which
  // minimize the sum of all squared residuals.
  // f(ind) = (a-b*ind)^(-1)
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
  functor.m = vecerror.size(); // num_pixels;
  functor.n = n;

  Eigen::LevenbergMarquardt<LMFunctor, double> lm(functor);
  // int status = lm.minimize(params);

  /*
  // ----- start: for unit test - SHOULD REMOVE ------
  std::cout << "LM optimization status: " << status << std::endl;
  std::cout << "LM optimization iterations: " << lm.iter << std::endl;
  std::cout << "estimated parameters: "
            << "\ta: " << params(0) << "\tb: " << params(1) << std::endl;

  Eigen::VectorXd gt_params(n);
  gt_params(0) = 2.f;
  gt_params(1) = 2.f;
  std::cout << "ground-truth parameters: "
            << "\ta: " << gt_params(0) << "\t\tb: " << gt_params(1)
            << std::endl;
  // ----- end: for unit test - SHOULD REMOVE ------
  */

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

  // get the kneepoint index
  // we need to multply by the number of pixels to undo the normalization
  int kneepointIndex = std::max<int>(
      0, std::min<int>(int(kneepoint * num_pixels), vecerror.size() - 1));
  std::cout << "Kneepoint index: " << kneepointIndex << std::endl;

  // the integral of the errors we can tolerate is in the measuredValues at the
  // kneepoint index we need to multiply by the total error to undo the
  // normalization
  std::cout << "Total error" << totalError << std::endl;
  std::cout << "Measured value" << measuredValues(kneepointIndex, 1)
            << std::endl;
  // errorBudget = measuredValues(kneepointIndex, 1) * totalError;
  errorBudget = measuredValues(kneepointIndex, 1) * patchSizeSquared;
  // errorBudget = measuredValues(kneepointIndex, 1) * meanError;

  /*
  // ----- start: for unit test - SHOULD REMOVE ------

  std::cout << "estimated knee point: " << kneepoint << std::endl;
  std::cout << "estimated error budget: " << errorBudget << std::endl;

  std::cout << "Note: should comment out these logs in "
               "ErrorBudgetCalculator.cpp in runtime)"
            << std::endl;

  // ----- end: for unit test - SHOULD REMOVE ------
  */

  return true;
}

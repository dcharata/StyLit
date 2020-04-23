#include "TestErrorBudget.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "Algorithm/ErrorBudgetCalculator.h"
#include "Algorithm/NNF.h"
#include "Algorithm/NNFError.h"
#include "CPU/ErrorBudgetCalculator.cpp"
#include "Configuration/Configuration.h"

using namespace std::chrono;

// ----------------------------------------------------------------------------------------
// unit test for the knee point finding functions
void generate_dummydata(int num_samples, Eigen::MatrixXd &measuredValues,
                        Eigen::VectorXd params, bool addnoise, bool sort,
                        bool shuffle) {

  // create an error vector
  double rand_scale = 1;
  double x_scale = 1 / (double)num_samples;
  std::vector<double> vecerror;
  for (int i = 0; i < num_samples; i++) {
    // hyperbolic function value
    double value = powf(params(0) - i * x_scale * params(1), -1);
    if (addnoise == true) {
      // add random noise
      value += rand_scale * (static_cast<double>(std::rand()) /
                             static_cast<double>(RAND_MAX));
    }
    vecerror.push_back(value);
  }

  if (sort == true) {
    // sort the data samples into accending order
    std::sort(vecerror.begin(), vecerror.end());
  }

  if (shuffle == true) {
    // Fisher-Yates shuffle
    // https://www.techiedelight.com/shuffle-given-array-elements-fisher-yates-shuffle/
    for (int i = 0; i < num_samples; i++) {
      // generate a random number j such that i<=j <n and
      // swap the element present at index j with the element
      // present at current index i
      int j = i + std::rand() % (num_samples - i);
      std::swap(vecerror[i], vecerror[j]);
    }
  }

  // convert to data_samples
  for (int i = 0; i < num_samples; i++) {
    measuredValues(i, 0) = i * x_scale;
    measuredValues(i, 1) = vecerror[i];
  }
}

void generate_errorimage(NNFError &nnferror, Eigen::VectorXd params,
                         bool addnoise, bool shuffle) {

  int height = nnferror.error.dimensions.rows;
  int width = nnferror.error.dimensions.cols;
  int num_pixels = height * width;
  Eigen::MatrixXd measuredValues(num_pixels, 2);
  bool sort = false;
  generate_dummydata(num_pixels, measuredValues, params, addnoise, sort,
                     shuffle);

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int i = row * width + col;
      nnferror.error(row, col)[0] = measuredValues(i, 1);
    }
  }
}

void test_hyperbolic_fitting(int num_pixels) {
  const int n = 2;
  Eigen::MatrixXd measuredValues(num_pixels, 2);
  Eigen::VectorXd gt_params(n);
  gt_params(0) = 2.f;
  gt_params(1) = 2.f;
  const bool addnoise = false;
  const bool sort = false;
  const bool shuffle = false;
  generate_dummydata(num_pixels, measuredValues, gt_params, addnoise, sort,
                     shuffle);

  // initilization
  Eigen::VectorXd params(n);
  params(0) = 1.0; // a
  params(1) = 1.0; // b

  LMFunctor functor;
  functor.measuredValues = measuredValues;
  functor.m = num_pixels;
  functor.n = n;

  Eigen::LevenbergMarquardt<LMFunctor, double> lm(functor);
  int status = lm.minimize(params);
  std::cout << "LM optimization status: " << status << std::endl;
  std::cout << "LM optimization iterations: " << lm.iter << std::endl;
  std::cout << "estimated parameters: "
            << "\ta: " << params(0) << "\tb: " << params(1) << std::endl;
  std::cout << "ground-truth parameters: "
            << "\ta: " << gt_params(0) << "\tb: " << gt_params(1) << std::endl;
}

// ----------------------------------------------------------------------------------------

bool TestErrorBudget::run() {
  std::cout << "Testing error budget calculation... " << std::endl;
  std::cout << std::endl;

  // Test curve fitting without noise
  std::cout << "1 - hyperbolic function fitting " << std::endl;
  const int num_pixels = 100;
  std::cout << "num_samples: " << num_pixels << std::endl;
  // runtime
  auto start = high_resolution_clock::now();
  test_hyperbolic_fitting(num_pixels);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "runtime: " << duration.count() << " milliseconds" << std::endl;
  std::cout << std::endl;

  // Test error budget
  // generate dummy nnferror data
  std::cout << "2 - error budget calculation " << std::endl;
  const int height = 600;
  const int width = 800;
  std::cout << "num_pixels: height " << height << " * width " << width
            << std::endl;
  const NNF nnf(ImageDimensions(height, width), ImageDimensions(height, width));
  NNFError nnferror = {nnf};
  float errorBudget = 0.f;

  // set gt hyperbolic function parameter
  Eigen::VectorXd gt_params(2);
  gt_params(0) = 2.f;
  gt_params(1) = 2.f;
  const bool addnoise = true;
  const bool shuffle = true;
  generate_errorimage(nnferror, gt_params, addnoise, shuffle);
  Configuration configuration;
  ErrorBudgetCalculator calc;
  // runtime
  start = high_resolution_clock::now();
  calc.calculateErrorBudget(configuration, nnferror, errorBudget);
  stop = high_resolution_clock::now();
  duration = duration_cast<milliseconds>(stop - start);
  std::cout << "runtime: " << duration.count() << " milliseconds" << std::endl;
  std::cout << std::endl;

  return true;
}

// NOTES:
/* use numerical jacobian of errors
 * eigen lib optimization seems slower than dlib
 * the nnferror error image is set to sourceDimensions at the moment (in
 * NNFError.cpp) the iterative optimization is probably not real time for a
 * large error image.
 */

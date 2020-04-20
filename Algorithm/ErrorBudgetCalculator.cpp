#include "ErrorBudgetCalculator.h"
#include <vector>
#include <bits/stdc++.h>
#include "NNFError.h"

// ----------------------------------------------------------------------------------------
// hyperbolic function fitting

double model(const input_vector& x, const parameter_vector& params) {
    // the hyperbolic function model
    // We will use this function to generate data.  It represents a function of 1 variable
    // and 2 parameters.   The least squares procedure will be used to infer the values of
    // the 2 parameters based on a set of input/output pairs.
    const double a = params(0);
    const double b = params(1);
    return powf(a-b*x, -1);
}

double residual(const std::pair<input_vector, double>& data,
               const parameter_vector& params) {
    // This function is the "residual" for a least squares problem.   It takes an input/output
    // pair and compares it to the output of our model and returns the amount of error.  The idea
    // is to find the set of parameters which makes the residual small on all the data pairs.
    return model(data.first, params) - data.second;
}

parameter_vector residual_derivative (const std::pair<input_vector, double>& data,
                          const parameter_vector& params) {
    // This function is the derivative of the residual() function with respect to the parameters.
    parameter_vector der;

    const double a = params(0);
    const double b = params(1);

    double temp = powf(a - b * data.first, -2);
    der(0) = -1.f * temp; // da
    der(1) = data.first * temp; // db

    return der;
}

bool comparator(const std::pair<std::pair<int, int>, double> lhs, const std::pair<std::pair<int, int>, float> rhs) {
    return lhs.second <rhs.second;
}

// ----------------------------------------------------------------------------------------

bool ErrorBudgetCalculator::calculateErrorBudget(
    const Configuration &configuration, NNFError &error,
    float errorBudget) {
  return implementationOfCalculateErrorBudget(configuration, error,
                                              errorBudget);
}

bool ErrorBudgetCalculator::implementationOfCalculateErrorBudget(
        const Configuration &configuration, NNFError &nnferror, float errorBudget) {
    // we may not need configuration?

    // read from the error image
    std::vector<std::pair<std::pair<int, int>, double>> vecerror;
    int height = nnferror.error.dimensions.rows;
    int width = nnferror.error.dimensions.cols;
    for (int row=0; row<height; row++) {
        for (int col=0; col<width; col++) {
            vecerror.push_back(std::make_pair(std::make_pair(row, col), (double)nnferror.error.data[row*width+col][0]));
        }
    }

    // sort the error vector
    sort(vecerror.begin(), vecerror.end(), &comparator);

    // convert to data_samples
    std::vector<std::pair<input_vector, double> > data_samples;
    double x_scale = 1.f / (height*width);
    for (int row=0; row<height; row++) {
        for (int col=0; col<width; col++) {
            int i = row * width + col;
            data_samples.push_back(std::make_pair(i*x_scale, vecerror[i].second));
            nnferror.errorIndex(vecerror[i].first.first, vecerror[i].first.second)[0] = i;
        }
    }

    // fit the hyperbolic function
    // use the Levenberg-Marquardt method to determine the parameters which
    // minimize the sum of all squared residuals.
    parameter_vector params;
    params = 1.f; // initialization
    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose(),
                           residual,
                           residual_derivative,
                           data_samples,
                           params);
    std::cout << "Note: should comment out these logs in ErrorBudgetCalculator.cpp in runtime)" << std::endl;
    std::cout << "estimated parameters: " << trans(params);
    parameter_vector gt_params = {2.f, 2.f};
    std::cout << "solution error: " << length(params - gt_params) << std::endl;
    std::cout << std::endl;

    // calculate the knee point
    double a = params(0);
    double b = params(1);
    int kneepoint = (int)sqrtf(1.f/b) + a/b;
    errorBudget = vecerror[kneepoint].second;
    std::cout << "estimated knee point: " << kneepoint << std::endl;
    std::cout << "estimated error budget: " << errorBudget << std::endl;

    return true;
}

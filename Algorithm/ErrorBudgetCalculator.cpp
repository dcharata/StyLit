#include "ErrorBudgetCalculator.h"
#include <vector>
#include <bits/stdc++.h>
#include "NNFError.h"
#include "Image.h"
#include "ImageDimensions.h"
#include "NNF.h"


// ----------------------------------------------------------------------------------------
// hyperbolic function fitting

double model(const input_vector& index, const parameter_vector& params) {
    // the hyperbolic function model
    // We will use this function to generate data.  It represents a function of 1 variable
    // and 2 parameters.   The least squares procedure will be used to infer the values of
    // the 2 parameters based on a set of input/output pairs.
    const double a = params(0);
    const double b = params(1);
    return powf(a-b*index, -1);
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



// ----------------------------------------------------------------------------------------

bool ErrorBudgetCalculator::calculateErrorBudget(
    const Configuration &configuration, const NNFError &error,
    float errorBudget) {
  return implementationOfCalculateErrorBudget(configuration, error,
                                              errorBudget);
}

bool ErrorBudgetCalculator::implementationOfCalculateErrorBudget(
        const Configuration &configuration, const NNFError &error, float errorBudget) {
//    // we may not need configuration?

//    // error the 2D array of errors used to calculate the knee point
//    // of size image height and image width?
//    std::vector<float> vecerror;
//    int height = error.dimensions.rows;
//    int width = error.dimensions.cols;
//    for (int row=0; row<height; row++) {
//        for (int col=0; col<width; col++) {
//            vecerror.push_back(error->operator(row, col));
//        }
//    }

//    // sort the error vector
//    sort(vecerror.begin(), vecerror.end());

//    // convert to data_samples
//    std::vector<std::pair<input_vector, float> > data_samples;
//    for (int i=0; i<height*width; i++) {
//        data_samples.push_back(std::make_pair((float)i, vecerror[i]));
//    }

//    // fit the hyperbolic function
//    // Use Levenberg-Marquardt
//    // optional alternatives:

//    // Use the Levenberg-Marquardt method to determine the parameters which
//    // minimize the sum of all squared residuals.
//    parameter_vector params;
//    params = 1;
//    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose(),
//                           residual,
//                           residual_derivative,
//                           data_samples,
//                           params);

//    float a = params(0);
//    float b = params(1);
//    errorBudget = sqrtf(1.f/b) + a/b;
//    std::cout << "wait" << std::endl;
//    test_errorbudget();

    return true;
}

#include "ErrorBudgetCalculator.h"

#include <vector>
#include <iostream>
#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>
#include "NNFError.h"

// ----------------------------------------------------------------------------------------
// hyperbolic function fitting

//double model(const input_vector& x, const parameter_vector& params) {
//    // the hyperbolic function model
//    // We will use this function to generate data.  It represents a function of 1 variable
//    // and 2 parameters.   The least squares procedure will be used to infer the values of
//    // the 2 parameters based on a set of input/output pairs.
//    const double a = params(0);
//    const double b = params(1);
//    return powf(a-b*x, -1);
//}

//double residual(const std::pair<input_vector, double>& data,
//               const parameter_vector& params) {
//    // This function is the "residual" for a least squares problem.   It takes an input/output
//    // pair and compares it to the output of our model and returns the amount of error.  The idea
//    // is to find the set of parameters which makes the residual small on all the data pairs.
//    return model(data.first, params) - data.second;
//}

//parameter_vector residual_derivative (const std::pair<input_vector, double>& data,
//                          const parameter_vector& params) {
//    // This function is the derivative of the residual() function with respect to the parameters.
//    parameter_vector der;

//    const double a = params(0);
//    const double b = params(1);

//    double temp = powf(a - b * data.first, -2);
//    der(0) = -1.f * temp; // da
//    der(1) = data.first * temp; // db

//    return der;
//}

struct LMFunctor
{
    // 'm' pairs of (x, f(x))
    Eigen::MatrixXf measuredValues;

    // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
    int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
    {
        // 'x' has dimensions n x 1
        // It contains the current estimates for the parameters.

        // 'fvec' has dimensions m x 1
        // It will contain the error for each data point.

        float aParam = x(0);
        float bParam = x(1);

        for (int i = 0; i < values(); i++) {
            float xValue = measuredValues(i, 0);
            float yValue = measuredValues(i, 1);

            fvec(i) = yValue - powf(aParam - bParam * xValue, -1);
        }
        return 0;
    }

    // Compute the jacobian of the errors
    int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const
    {
        // 'x' has dimensions n x 1
        // It contains the current estimates for the parameters.

        // 'fjac' has dimensions m x n
        // It will contain the jacobian of the errors, calculated numerically in this case.

        float epsilon;
        epsilon = 1e-7f;

        for (int i = 0; i < x.size(); i++) {
            // numerical Jacobian
            Eigen::VectorXf xPlus(x);
            xPlus(i) += epsilon;
            Eigen::VectorXf xMinus(x);
            xMinus(i) -= epsilon;

            Eigen::VectorXf fvecPlus(values());
            operator()(xPlus, fvecPlus);

            Eigen::VectorXf fvecMinus(values());
            operator()(xMinus, fvecMinus);

            Eigen::VectorXf fvecDiff(values());
            fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

            fjac.block(0, i, values(), 1) = fvecDiff;
        }

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

bool comparator(const std::pair<std::pair<int, int>, float> lhs, const std::pair<std::pair<int, int>, float> rhs) {
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
    std::vector<std::pair<std::pair<int, int>, float>> vecerror;
    int height = nnferror.error.dimensions.rows;
    int width = nnferror.error.dimensions.cols;
    int num_pixels = height * width;
    for (int row=0; row<height; row++) {
        for (int col=0; col<width; col++) {
            vecerror.push_back(std::make_pair(std::make_pair(row, col), nnferror.error.data[row*width+col][0]));
        }
    }

    // sort the error vector
    sort(vecerror.begin(), vecerror.end(), &comparator);

    // convert to eigen matrix
    // ref: https://medium.com/@sarvagya.vaish/levenberg-marquardt-optimization-part-2-5a71f7db27a0
    Eigen::MatrixXf measuredValues(num_pixels, 2); // pairs of (x, f(x))
    float x_scale = 1.f / (height*width);
    for (int row=0; row<height; row++) {
        for (int col=0; col<width; col++) {
            int i = row * width + col;
            measuredValues(i, 0) = i*x_scale;
            measuredValues(i,1) = vecerror[i].second;
            nnferror.errorIndex(vecerror[i].first.first, vecerror[i].first.second)[0] = i;
        }
    }

    // fit the hyperbolic function
    // use the Levenberg-Marquardt method to determine the parameters which
    // minimize the sum of all squared residuals.
    // f(ind) = (a-b*ind)^(-1)
    int n = 2; // number of parameters
    // 'x' is vector of length 'n' containing the initial values for the parameters.
    Eigen::VectorXf params(n);
    // initialization
    params(0) = 1.f; // a
    params(1) = 1.f; // b

    // Run the LM optimization
    // Create a LevenbergMarquardt object and pass it the functor.
    LMFunctor functor;
    functor.measuredValues = measuredValues;
    functor.m = num_pixels;
    functor.n = n;

    Eigen::LevenbergMarquardt<LMFunctor, float> lm(functor);
    int status = lm.minimize(params);
    std::cout << "LM optimization status: " << status << std::endl;
    std::cout << "estimated parameters: " << "\ta: " << params(0) << "\tb: " << params(1) << std::endl;

    std::cout << "Note: should comment out these logs in ErrorBudgetCalculator.cpp in runtime)" << std::endl;
    Eigen::VectorXf gt_params(n);
    gt_params(0) = 2.f;
    gt_params(1) = 2.f;
    std::cout << "ground-truth parameters: " << "\ta: " << gt_params(0) << "\tb: " << gt_params(1) << std::endl;

    // calculate the knee point
    float a = params(0);
    float b = params(1);
    int kneepoint = (int)sqrtf(1.f/b) + a/b;
    errorBudget = vecerror[kneepoint].second;
    std::cout << "estimated knee point: " << kneepoint << std::endl;
    std::cout << "estimated error budget: " << errorBudget << std::endl;

    return true;
}

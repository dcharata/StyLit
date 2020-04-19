#include "TestErrorBudget.h"

#include <vector>
#include <bits/stdc++.h>
//#define DLIB_NO_GUI_SUPPORT
#include <dlib/optimization.h>

#include "Algorithm/ErrorBudgetCalculator.h"
#include "Algorithm/ErrorBudgetCalculator.cpp"

using namespace dlib;

// ----------------------------------------------------------------------------------------
// unit test for the knee point finding functions
void test_generate_datasamples(
        std::vector<std::pair<input_vector, double>>& data_samples,
        parameter_vector params,
        bool addnoise, bool sort) {
    // create an error vector
    int height = 600; // dummy
    int width = 800;

    double rand_scale = 1.f;
    double x_scale = 1.f / (height*width);
    std::vector<double> vecerror;
    for (double i=0.f; i<height*width; i++) {
        // hyperbolic function value
        double value = model(i*x_scale, params);
        if (addnoise==true) {
            // add random noise
            value += rand_scale * (static_cast <double> (std::rand()) / static_cast <double> (RAND_MAX));
        }
        vecerror.push_back(value);
    }

//    if (sort==true) {
//        // sort the data samples into accending order
//        std::sort( vecerror.begin(), vecerror.end() );
//    }


    // convert to data_samples
    for (int i=0; i<height*width; i++) {
        data_samples.push_back(std::make_pair(i*x_scale, vecerror[i]));
    }
}

void test_hyperbolic_derivative(
        const std::pair<input_vector, double>& data,
        const parameter_vector params) {
    // Let's make sure that our derivative function defined above matches
    // the approximate derivative computed using central differences (via derivative()).
    // If this value is big then it means we probably typed the derivative function incorrectly.
    std::cout << "derivative error: " <<
                 length(residual_derivative(data, params) -
                   derivative(residual)(data, params)) << std::endl;
}

void test_errorbudget() {
    std::vector<std::pair<input_vector, double> > data_samples;
    parameter_vector gt_params = {2.f, 2.f}; // a, b
    std::cout << "ground-truth parameters: " << trans(gt_params) << std::endl;

    bool addnoise = true;
    bool sort = false;
    test_generate_datasamples(data_samples, gt_params, addnoise, sort);

//    test_hyperbolic_derivative(data_samples[0], gt_params);

    // optimization - 3 different methods
    // to be determined with real error data
    // ref: http://dlib.net/least_squares_ex.cpp.html
    parameter_vector params;

    // Use the Levenberg-Marquardt method to determine the parameters which
    // minimize the sum of all squared residuals.
    std::cout << "Use Levenberg-Marquardt" << std::endl;
    params = 1.f; // initilization
    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose(),
                           residual,
                           residual_derivative,
                           data_samples,
                           params);
    std::cout << "estimated parameters: " << trans(params)<< std::endl;
    std::cout << "solution error: " << length(params - gt_params) << std::endl;
    std::cout << std::endl;

    // If we didn't create the residual_derivative function then we could
    // have used this method which numerically approximates the derivatives.
    std::cout << "Use Levenberg-Marquardt, approximate derivatives" << std::endl;
    params = 1.f; // initilization
    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose(),
                           residual,
                           derivative(residual),
                           data_samples,
                           params);
    std::cout << "estimated parameters: " << trans(params)<< std::endl;
    std::cout << "solution error: " << length(params - gt_params) << std::endl;
    std::cout << std::endl;

    // This version of the solver uses a method which is appropriate for problems
    // where the residuals don't go to zero at the solution.  So in these cases
    // it may provide a better answer.
    std::cout << "Use Levenberg-Marquardt/quasi-newton hybrid" << std::endl;
    params = 1.f; // initilization
    solve_least_squares(objective_delta_stop_strategy(1e-7).be_verbose(),
                           residual,
                           residual_derivative,
                           data_samples,
                           params);
    std::cout << "estimated parameters: " << trans(params)<< std::endl;
    std::cout << "solution error: " << length(params - gt_params) << std::endl;
    std::cout << std::endl;
}


// ----------------------------------------------------------------------------------------

bool TestErrorBudget::run()
{
    std::cout << "Test error budget... " << std::endl;
    test_errorbudget();
    return true;
}


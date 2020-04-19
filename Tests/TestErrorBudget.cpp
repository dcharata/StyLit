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
        bool addnoise, bool shuffle) {
    // create an error vector
    int height = 600; // dummy
    int width = 800;

    double rand_scale = 1.f;
    double index_scale = 1.f / (height*width);
    std::vector<double> vecerror;
    for (double i=0.f; i<height*width; i++) {
        // hyperbolic function value
        double value = model(i*index_scale, params);
        if (addnoise==true) {
            // add random noise
            value += rand_scale * (static_cast <double> (std::rand()) / static_cast <double> (RAND_MAX));
        }
        vecerror.push_back(value);
    }

//    if (shuffle==true) {
//        // shuffle the vector
//        std::random_shuffle ( vecerror.begin(), vecerror.end() );
//    }


    // convert to data_samples
    for (int i=0; i<height*width; i++) {
        data_samples.push_back(std::make_pair(i*index_scale, vecerror[i]));
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
    parameter_vector gt_params = {1.f, 1.f}; // a, b
    std::cout << "ground-truth parameters: " << trans(gt_params) << std::endl;

    bool addnoise = true;
    bool shuffle = false;
    test_generate_datasamples(data_samples, gt_params, addnoise, shuffle);

    test_hyperbolic_derivative(data_samples[0], gt_params);

    // ref: http://dlib.net/least_squares_ex.cpp.html
    // Use the Levenberg-Marquardt method to determine the parameters which
    // minimize the sum of all squared residuals.
    parameter_vector params;
    params = 0.5;
    solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose(),
                           residual,
//                           derivative(residual),
                           residual_derivative,
                           data_samples,
                           params);

    std::cout << "estimated parameters: " << trans(params)<< std::endl;
    std::cout << "solution error: " << length(params - gt_params) << std::endl;
}


// ----------------------------------------------------------------------------------------

bool TestErrorBudget::run()
{
    std::cout << "Test error budget... " << std::endl;
    test_errorbudget();
    return true;
}


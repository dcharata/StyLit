#include "TestErrorBudget.h"

#include <vector>
#include <iostream>
//#define DLIB_NO_GUI_SUPPORT
#include <dlib/optimization.h>

#include "Algorithm/NNF.h"
#include "Algorithm/NNFError.h"
#include "Configuration/Configuration.h"
#include "Algorithm/ErrorBudgetCalculator.h"
#include "Algorithm/ErrorBudgetCalculator.cpp"

using namespace dlib;

// ----------------------------------------------------------------------------------------
// unit test for the knee point finding functions
void generate_datasamples(
        int num_samples,
        std::vector<std::pair<input_vector, double>>& data_samples,
        parameter_vector params,
        bool addnoise, bool sort, bool shuffle) {

    // create an error vector
    double rand_scale = 1.f;
    double x_scale = 1.f / num_samples;
    std::vector<double> vecerror;
    for (double i=0.f; i<num_samples; i++) {
        // hyperbolic function value
        double value = model(i*x_scale, params);
        if (addnoise==true) {
            // add random noise
            value += rand_scale * (static_cast <double> (std::rand()) / static_cast <double> (RAND_MAX));
        }
        vecerror.push_back(value);
    }

    if (sort==true) {
        // sort the data samples into accending order
        std::sort( vecerror.begin(), vecerror.end() );
    }

    if (shuffle==true) {
        // Fisher-Yates shuffle
        // https://www.techiedelight.com/shuffle-given-array-elements-fisher-yates-shuffle/
        for(int i=0; i<num_samples; i++) {
            // generate a random number j such that i<=j <n and
            // swap the element present at index j with the element
            // present at current index i
            int j = i + std::rand() % (num_samples - i);
            std::swap(vecerror[i], vecerror[j]);
        }
    }

    // convert to data_samples
    for (int i=0; i<num_samples; i++) {
        data_samples.push_back(std::make_pair(i*x_scale, vecerror[i]));
    }
}

void generate_errorimage(NNFError &nnferror, parameter_vector params, bool addnoise, bool shuffle) {
    std::vector<std::pair<input_vector, double> > data_samples;
    int height = nnferror.error.dimensions.rows;
    int width = nnferror.error.dimensions.cols;
    int num_samples = height * width;
    bool sort = false;
    generate_datasamples(num_samples, data_samples, params, addnoise, sort, shuffle);

    for (int row=0; row<height; row++) {
        for (int col=0; col<width; col++) {
            int i = row * width + col;
            nnferror.error(row, col)[0] = (float)data_samples[i].second;
        }
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

void test_hyperbolic_fitting() {
    std::vector<std::pair<input_vector, double> > data_samples;
    parameter_vector gt_params = {2.f, 2.f}; // a, b
    std::cout << "ground-truth parameters: " << trans(gt_params) << std::endl;

    int num_samples = 10000;
    bool addnoise = true;
    bool sort = false;
    bool shuffle = false;
    generate_datasamples(num_samples, data_samples, gt_params, addnoise, sort, shuffle);

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
    std::cout << "estimated parameters: " << trans(params);
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
    std::cout << "estimated parameters: " << trans(params);;
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
    std::cout << "estimated parameters: " << trans(params);
    std::cout << "solution error: " << length(params - gt_params) << std::endl;
    std::cout << std::endl;
}


// ----------------------------------------------------------------------------------------

bool TestErrorBudget::run()
{
    std::cout << "Testing error budget... " << std::endl;
    std::cout << std::endl;

    // curve fitting
//    test_hyperbolic_fitting();

    // error budget
    // Generate dummy nnferror data
    int height = 600;
    int width = 800;
    const NNF nnf(ImageDimensions(height, width), ImageDimensions(height, width));
    NNFError nnferror = {nnf};
    float errorBudget = 0.f;

    // set gt hyperbolic function parameter
    parameter_vector gt_params = {2.f, 2.f}; // a, b
    std::cout << "ground-truth parameters: " << trans(gt_params) << std::endl;
    bool addnoise = true;
    bool shuffle = true;
    generate_errorimage(nnferror, gt_params, addnoise, shuffle);
    Configuration configuration;
    ErrorBudgetCalculator calc;
    calc.calculateErrorBudget(configuration, nnferror, errorBudget);

    return true;
}

// NOTES:
/* not sure if the objective of the first solver should be 0 in the end of optimization
 * should test the solvers on real data and pick one
 * the nnferror error image is set to sourceDimensions at the moment (in NNFError.cpp)
 * the optimization is probably not real time for a large error image.
*/

#include "TestErrorBudget.h"

#include <vector>
#include <iostream>

#include "Algorithm/NNF.h"
#include "Algorithm/NNFError.h"
#include "Configuration/Configuration.h"
#include "Algorithm/ErrorBudgetCalculator.h"
#include "Algorithm/ErrorBudgetCalculator.cpp"

// ----------------------------------------------------------------------------------------
// unit test for the knee point finding functions
void generate_dummydata(
        int num_samples,
        Eigen::MatrixXf& measuredValues,
        Eigen::VectorXf params,
        bool addnoise, bool sort, bool shuffle) {

    // create an error vector
    float rand_scale = 1.f;
    float x_scale = 1.f / num_samples;
    std::vector<float> vecerror;
    for (float i=0.f; i<num_samples; i++) {
        // hyperbolic function value
        float value = powf(params(0) - i*x_scale * params(1), -1);
        if (addnoise==true) {
            // add random noise
            value += rand_scale * (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX));
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
        measuredValues(i, 0) = i*x_scale;
        measuredValues(i, 1) = vecerror[i];
    }
}

void generate_errorimage(NNFError &nnferror, Eigen::VectorXf params, bool addnoise, bool shuffle) {

    int height = nnferror.error.dimensions.rows;
    int width = nnferror.error.dimensions.cols;
    int num_pixels = height * width;
    Eigen::MatrixXf measuredValues(num_pixels, 2);
    bool sort = false;
    generate_dummydata(num_pixels, measuredValues, params, addnoise, sort, shuffle);

    for (int row=0; row<height; row++) {
        for (int col=0; col<width; col++) {
            int i = row * width + col;
            nnferror.error(row, col)[0] = measuredValues(i,1);
        }
    }
}

void test_hyperbolic_fitting() {
    int num_pixels = 10000;
    int n = 2;
    Eigen::MatrixXf measuredValues(num_pixels, 2);
    Eigen::VectorXf gt_params(n);
    gt_params(0) = 2.f;
    gt_params(1) = 2.f;
    bool addnoise = true;
    bool sort = false;
    bool shuffle = false;
    generate_dummydata(num_pixels, measuredValues, gt_params, addnoise, sort, shuffle);

    Eigen::VectorXf params(n);
    params(0) = 1.f; // a
    params(1) = 1.f; // b

    LMFunctor functor;
    functor.measuredValues = measuredValues;
    functor.m = num_pixels;
    functor.n = n;

    Eigen::LevenbergMarquardt<LMFunctor, float> lm(functor);
    int status = lm.minimize(params);
    std::cout << "LM optimization status: " << status << std::endl;
    std::cout << "estimated parameters: " << "\ta: " << params(0) << "\tb: " << params(1) << std::endl;
    std::cout << "ground-truth parameters: " << "\ta: " << gt_params(0) << "\t\tb: " << gt_params(1) << std::endl;
}


// ----------------------------------------------------------------------------------------

bool TestErrorBudget::run()
{
    std::cout << "Testing error budget... " << std::endl;
    std::cout << std::endl;

    // curve fitting
    test_hyperbolic_fitting();

//    // error budget
//    // Generate dummy nnferror data
//    int height = 600;
//    int width = 800;
//    const NNF nnf(ImageDimensions(height, width), ImageDimensions(height, width));
//    NNFError nnferror = {nnf};
//    float errorBudget = 0.f;

//    // set gt hyperbolic function parameter
//    Eigen::VectorXf gt_params;
//    gt_params(0) = 2.f;
//    gt_params(1) = 2.f;
//    std::cout << "ground-truth parameters: " << "\ta: " << gt_params(0) << "\tb: " << gt_params(1) << std::endl;
//    bool addnoise = true;
//    bool shuffle = true;
//    generate_errorimage(nnferror, gt_params, addnoise, shuffle);
//    Configuration configuration;
//    ErrorBudgetCalculator calc;
//    calc.calculateErrorBudget(configuration, nnferror, errorBudget);

    return true;
}

// NOTES:
/* use numerical jacobian of errors
 * estimated parameter has a larger error that using dlib
 * the nnferror error image is set to sourceDimensions at the moment (in NNFError.cpp)
 * the iterative optimization is probably not real time for a large error image.
*/

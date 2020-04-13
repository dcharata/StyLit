#ifndef FEATUREVECTOR_H
#define FEATUREVECTOR_H

#include <Eigen/Dense>

// A FeatureVector is just a wrapper around an Eigen vector.
template <typename T, unsigned int numChannels>
using FeatureVector = Eigen::Matrix<T, numChannels, 1>;

#endif // FEATUREVECTOR_H

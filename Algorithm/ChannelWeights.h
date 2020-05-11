#ifndef CHANNELWEIGHTS_H
#define CHANNELWEIGHTS_H

#include <Eigen/Dense>

// A ChannelWeights vector is just a wrapper around an Eigen vector.
template <unsigned int numChannels> using ChannelWeights = Eigen::Matrix<float, numChannels, 1>;

#endif // CHANNELWEIGHTS_H

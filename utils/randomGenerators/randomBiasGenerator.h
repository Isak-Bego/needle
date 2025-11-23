#ifndef RANDOMBIASGENERATOR_H
#define RANDOMBIASGENERATOR_H

#include <random>

/**
*
* @brief Helper function for generating a random bias for the neurons
*
* @returns - A randomized value for the bias
*/
inline double generate_bias() {
    std::random_device rd;
    std::mt19937 gen(rd());

    double mean = 0.0;
    double stddev = 1.0;
    std::normal_distribution<double> dist(mean, stddev);

    return dist(gen);
}

#endif //RANDOMBIASGENERATOR_H

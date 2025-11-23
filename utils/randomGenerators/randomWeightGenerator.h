#ifndef RANDOMWEIGHTGENERATOR_H
#define RANDOMWEIGHTGENERATOR_H

#include <random>
#include <cmath>

/**
* According to Michael Nielsen's book on Neural networks, which you can access through: http://neuralnetworksanddeeplearning.com/
* Since the activation of the neuron is calculated by running the weighted sum through a sigmoid function which compresses this sum
* in the range [0, 1], if the number is too big, we will end up having an activation of the neuron that is close to 1, which will lead
* to a very small partial gradient (Check the formula in sigmoidNode.h). A small partial gradient means a smaller influence
* on the minimization of the loss function and consequently an increased learning time, which is not beneficial to us.
*
* @returns - A randomized value for the weight
*/


inline double generate_weight(int numberOfInputs) {
    std::random_device rd;
    std::mt19937 gen(rd());

    double mean = 0.0;
    double stddev = 1.0 / sqrt(numberOfInputs);
    std::normal_distribution<double> dist(mean, stddev);

    return dist(gen);
}

#endif //RANDOMWEIGHTGENERATOR_H

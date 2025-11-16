//
// According to Michael Nielsen's book on Neural networks. TODO: Add reference
// This is the optimal way for initalizing the weights so that we lower the chances of ending up in a
// saturated neuron which would greatly affect the speed of learning. The activation of the neuron is caluclated by
// running the weighted sum through a sigmoid function which compresses this sum in the range [0, 1]. Past a certain
// point, if the number is too big, we will end up having an activation of the neuron that is close to 1, thus greatly
// affecting the learning speed.
//

#ifndef RANDOMWEIGHTGENERATOR_H
#define RANDOMWEIGHTGENERATOR_H
#include <random>
#include <cmath>

inline double generate_weight(int numberOfWeights) {
    // Step 1: Create a random number generator (engine)
    std::random_device rd; // non-deterministic seed
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Step 2: Define a normal (Gaussian) distribution
    double mean = 0.0; // μ
    double stddev = 1.0 / sqrt(numberOfWeights); // σ
    std::normal_distribution<double> dist(mean, stddev);

    return dist(gen);
}

#endif //RANDOMWEIGHTGENERATOR_H

#ifndef NEURON_H
#define NEURON_H
#include <iostream>
#include "../utils/random-generators/randomBiasGenerator.h"

class Neuron {
    double activation = 0.0;
    std::vector<double> weights = std::vector<double>();
    double bias = static_cast<double>(generate_bias());

public:
    Neuron() = default;

    explicit Neuron(const double activation) {
        this->activation = activation;
    }

    explicit Neuron(const std::vector<double> &weights) {
        this->weights = weights;
    }

    void print() const {
        std::cout << "Activation: " << activation << std::endl;
        std::cout << "Bias: " << bias << std::endl;
        for (const auto &w : weights) {
            std::cout << w << std::endl;
        }
    }

    std::vector<double> &getWeights() { return this->weights; }
    void setWeights(std::vector<double> weights) { this->weights = weights; }
    double getBias() const{ return this->bias; }
    void setActivation(const double activation) { this->activation = activation; }
    double getActivation() const { return this->activation; }
};

#endif //NEURON_H

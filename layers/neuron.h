#ifndef NEURON_H
#define NEURON_H
#include <iostream>
#include "../utils/random-generators/randomBiasGenerator.h"

class Neuron {
    float activation = 0.0;
    std::vector<float> weights = std::vector<float>();
    float bias = static_cast<float>(generate_bias());

public:
    Neuron() = default;

    explicit Neuron(const float activation) {
        this->activation = activation;
    }

    explicit Neuron(const std::vector<float> &weights) {
        this->weights = weights;
    }

    void print() const {
        std::cout << "Activation: " << activation << std::endl;
        std::cout << "Bias: " << bias << std::endl;
        for (const auto &w : weights) {
            std::cout << w << std::endl;
        }
    }

    std::vector<float> &getWeights() { return this->weights; }
    void setWeights(std::vector<float> weights) { this->weights = weights; }
    float getBias() const{ return this->bias; }
    void setActivation(const float activation) { this->activation = activation; }
    float getActivation() const { return this->activation; }
};

#endif //NEURON_H

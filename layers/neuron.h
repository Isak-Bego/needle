#ifndef NEURON_H
#define NEURON_H
#include <iostream>
#include "../utils/random-generators/randomBiasGenerator.h"
#include "../utils/expressionNode.h"

class Neuron {
    Node activation = Node(0.0, false);
    std::vector<Node> weights;
    Node bias = Node(static_cast<double>(generate_bias()), false);

public:
    Neuron() = default;

    explicit Neuron(const double act) {
        this->setActivation(act);
    }

    explicit Neuron(const std::vector<double> &weights) {
        this->setWeights(weights);
    }

    void print() const {
        std::cout << "Activation: " << activation.get_value() << std::endl;
        std::cout << "Bias: " << bias.get_value() << std::endl;
        for (const Node &w: weights) {
            std::cout << w.get_value() << std::endl;
        }
    }

    std::vector<Node> &getWeights() { return this->weights; }
    void setWeights(const std::vector<double> &weights) {
        for (double weight : weights) {
            this->weights.emplace_back(weight, true);
        }
    }
    Node &getBias() { return this->bias; }
    void setActivation(const double &activation) { this->activation.set_value(activation); }
    Node *getActivation() { return &this->activation; }
};

#endif //NEURON_H

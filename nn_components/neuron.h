#ifndef NEURON_H
#define NEURON_H
#include <iostream>
#include <utils/random-generators/randomBiasGenerator.h>
#include <utils/node.h>

class Neuron {
    Node *activation = new Node(0.0, false);
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

    void print() {
        std::cout << "Activation: " << activation->get_value() << std::endl;
        std::cout << "Bias: " << bias.get_value() << std::endl;
        for (auto i = 0; i < weights.size(); i++) {
            std::cout << "W" << i << " : " << "Value: " << weights.at(i).get_value() << ", Gradient: " << weights.at(i).
                    get_gradient() << std::endl;
        }
        std::cout << std::endl;
    }

    std::vector<Node> &getWeights() { return this->weights; }

    void setWeights(const std::vector<double> &weights) {
        for (double weight: weights) {
            this->weights.emplace_back(weight, true);
        }
    }

    Node &getBias() { return this->bias; }
    void setActivation(const double &activation) const { this->activation->set_value(activation); }
    Node *getActivation() const { return this->activation; }
    void setActivationNode(Node *activation) { this->activation = activation; }
};

#endif //NEURON_H

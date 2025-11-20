#ifndef NEURON_H
#define NEURON_H
#include <random>
#include <nnComponents/module.h>
#include <nnComponents/activations/simoidNode.h>
#include "activations/relu.h"
#include "utils/randomGenerators/randomBiasGenerator.h"
#include "utils/randomGenerators/randomWeightGenerator.h"

enum class Activation {
    INPUT,
    RELU,
    SIGMOID,
    LINEAR,
    SOFTMAX
};

class Neuron final : public Module {
    std::vector<Node *> weights;
    Node *bias;
    Activation activation;

public:
    explicit Neuron(int numberOfInputs, const Activation act = Activation::RELU)
        : bias(new Node(generate_bias())), activation(act) {
        weights.reserve(numberOfInputs);
        for (int i = 0; i < numberOfInputs; ++i) {
            weights.push_back(new Node(generate_weight(numberOfInputs)));
        }
        bias = new Node(0.0);
    }

    Node *operator()(const std::vector<Node *> &inputVector) {
        Node *weightedSum = bias;

        for (size_t i = 0; i < weights.size(); ++i) {
            weightedSum = (*weightedSum) + *((*weights.at(i)) * (*inputVector.at(i)));
        }

        switch (activation) {
            case Activation::RELU:
                return relu(weightedSum);
            case Activation::SIGMOID:
                return sigmoid(weightedSum);
            default:
                return weightedSum;
        }
    }

    std::vector<Node *> parameters() override {
        std::vector<Node *> params = weights;
        params.push_back(bias);
        return params;
    }

    std::string representation() const {
        std::string act_str;
        switch (activation) {
            case Activation::RELU: act_str = "ReLU";
                break;
            case Activation::SIGMOID: act_str = "Sigmoid";
                break;
            case Activation::LINEAR: act_str = "Linear";
                break;
            case Activation::INPUT: act_str = "Input";
                break;
            case Activation::SOFTMAX: act_str = "Softmax";
                break;
        }
        return act_str + "Neuron(" + std::to_string(weights.size()) + ")";
    }
};

inline std::ostream &operator<<(std::ostream &os, const Neuron &n) {
    return os << n.representation();
}


#endif //NEURON_H

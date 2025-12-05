#ifndef NEURON_H
#define NEURON_H
#include <random>
#include <nnComponents/module.h>
#include <nnComponents/activations/sigmoidNode.h>
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

/**
 * Neurons get a set of inputs, they calculate the weighted sum and then run it through
 * an activation function to produce one of the elements of the output vector of a layer. They are the gears that
 * spin the mechanism of a Multi-Layered Perceptron. 
 */
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

    /**
     * @brief The core mechanism that is responsible for forward passing. 
     * 
     * @param inputVector - The input Nodes that are connected to the neuron, which in our case consist of all the neurons from the previous layer
     * @return - Returns the activation
     */
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

    /**
     * 
     * @return - Returns a list of parameters for a neuron which include all the weights and biases.
     */
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

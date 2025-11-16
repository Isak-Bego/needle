#ifndef NEURON_H
#define NEURON_H
#include <random>
#include <nnComponents/module.h>
#include <utils/activations/simoidNode.h>

enum class Activation {
    RELU,
    SIGMOID,
    LINEAR
};

class Neuron final : public Module {
    std::vector<Node*> weights;
    Node* bias;
    Activation activation;

public:
    explicit Neuron(int numberOfInputs, Activation act = Activation::RELU)
        : bias(nullptr), activation(act) {
        static thread_local std::mt19937 gen{std::random_device{}()};
        static thread_local std::uniform_real_distribution<double> dist(-1.0, 1.0);

        weights.reserve(numberOfInputs);
        for (int i = 0; i < numberOfInputs; ++i) {
            weights.push_back(new Node(dist(gen)));
        }
        bias = new Node(0.0);
    }

    Node* operator()(const std::vector<Node*>& inputVector) {
        Node* weightedSum = bias;

        for (size_t i = 0; i < weights.size(); ++i) {
            weightedSum = (*weightedSum) + *((*weights.at(i)) * (*inputVector.at(i)));
        }

        switch (activation) {
            case Activation::RELU:
                return weightedSum->relu();
            case Activation::SIGMOID:
                return sigmoid(weightedSum);
            case Activation::LINEAR:
                return weightedSum;
            default:
                return weightedSum;
        }
    }

    std::vector<Node*> parameters() override {
        std::vector<Node*> params = weights;
        params.push_back(bias);
        return params;
    }

    std::string representation() const {
        std::string act_str;
        switch (activation) {
            case Activation::RELU: act_str = "ReLU"; break;
            case Activation::SIGMOID: act_str = "Sigmoid"; break;
            case Activation::LINEAR: act_str = "Linear"; break;
        }
        return act_str + "Neuron(" + std::to_string(weights.size()) + ")";
    }
};

inline std::ostream& operator<<(std::ostream& os, const Neuron& n) {
    return os << n.representation();
}


#endif //NEURON_H

#ifndef NEURON_H
#define NEURON_H
#include <random>
#include <nn_components/module.h>

class Neuron final : public Module {
    std::vector<Node *> weights;
    Node *bias;
    bool isNonlinear;

public:
    explicit Neuron(int numberOfInputs, const bool isNonlinear = true)
        : bias(nullptr), isNonlinear(isNonlinear) {
        static thread_local std::mt19937 gen{std::random_device{}()};
        static thread_local std::uniform_real_distribution<double> dist(-1.0, 1.0);

        weights.reserve(numberOfInputs);
        for (int i = 0; i < numberOfInputs; ++i) {
            weights.push_back(new Node(dist(gen)));
        }
        bias = new Node(0.0);
    }

    Node *operator()(const std::vector<Node *> &inputVector) {
        Node *weightedSum = bias;

        for (size_t i = 0; i < weights.size(); ++i) {
            weightedSum = (*weightedSum) + *((*weights.at(i)) * (*inputVector.at(i)));
        }

        return isNonlinear ? weightedSum->relu() : weightedSum;
    }

    std::vector<Node *> parameters() override {
        std::vector<Node *> params = weights;
        params.push_back(bias);
        return params;
    }

    std::string representation() const {
        return (isNonlinear ? "ReLUNeuron(" : "LinearNeuron(") + std::to_string(weights.size()) + ")";
    }
};

inline std::ostream &operator<<(std::ostream &os, const Neuron &n) {
    return os << n.representation();
}

#endif //NEURON_H

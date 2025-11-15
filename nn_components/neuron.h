#ifndef NEURON_H
#define NEURON_H
#include <random>
#include <nn_components/module.h>

class Neuron final : public Module {
    std::vector<Node *> w; // weights
    Node *b; // bias
    bool nonlin;

public:
    explicit Neuron(int nin, bool nonlin = true)
        : b(nullptr), nonlin(nonlin) {
        static thread_local std::mt19937 gen{std::random_device{}()};
        static thread_local std::uniform_real_distribution<double> dist(-1.0, 1.0);

        w.reserve(nin);
        for (int i = 0; i < nin; ++i) {
            w.push_back(new Node(dist(gen)));
        }
        b = new Node(0.0);
    }

    Node *operator()(const std::vector<Node *> &x) {
        Node *act = b;

        for (size_t i = 0; i < w.size(); ++i) {
            act = (*act) + *((*w.at(i)) * (*x.at(i)));
        }

        return nonlin ? act->relu() : act;
    }

    std::vector<Node *> parameters() override {
        std::vector<Node *> params = w;
        params.push_back(b);
        return params;
    }

    std::string repr() const {
        return (nonlin ? "ReLUNeuron(" : "LinearNeuron(") + std::to_string(w.size()) + ")";
    }
};

inline std::ostream &operator<<(std::ostream &os, const Neuron &n) {
    return os << n.repr();
}

#endif //NEURON_H

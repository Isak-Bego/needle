#ifndef NETWORK_H
#define NETWORK_H
#include <nn_components/module.h>
#include <nn_components/layer.h>

class Network final : public Module {
    std::vector<Layer> layers;

public:
    Network(const int nin, const std::vector<int> &nouts) {
        std::vector<int> sz;
        sz.reserve(nouts.size() + 1);
        sz.push_back(nin);
        sz.insert(sz.end(), nouts.begin(), nouts.end());

        for (size_t i = 0; i < nouts.size(); ++i) {
            bool nonlin = (i != nouts.size() - 1);
            layers.emplace_back(sz.at(i), sz.at(i+1), nonlin);
        }
    }

    std::vector<Node *> operator()(const std::vector<Node *> &x_in) {
        std::vector<Node *> x = x_in;
        for (auto &layer: layers) {
            x = layer(x);
        }
        return x;
    }

    std::vector<Node *> parameters() override {
        std::vector<Node *> params;
        for (auto &layer: layers) {
            auto lp = layer.parameters();
            params.insert(params.end(), lp.begin(), lp.end());
        }
        return params;
    }

    std::string repr() const {
        std::string s = "Network of [";
        for (size_t i = 0; i < layers.size(); ++i) {
            s += layers.at(i).repr();
            if (i + 1 < layers.size()) s += ", ";
        }
        return s + "]";
    }
};

inline std::ostream &operator<<(std::ostream &os, const Network &m) {
    return os << m.repr();
}

#endif //NETWORK_H

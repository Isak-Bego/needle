#ifndef NETWORK_H
#define NETWORK_H
#include <nnComponents/module.h>
#include <nnComponents/layer.h>

class Network final : public Module {
    std::vector<Layer> layers;

public:
    Network(const int numberOfInputs, const std::vector<int> &numberOfOutputs) {
        // The vector below holds the dimensions of each layer of the network
        std::vector<int> networkDimensions;
        networkDimensions.reserve(numberOfOutputs.size() + 1);
        networkDimensions.push_back(numberOfInputs);
        networkDimensions.insert(networkDimensions.end(), numberOfOutputs.begin(), numberOfOutputs.end());

        for (size_t i = 0; i < numberOfOutputs.size(); ++i) {
            bool nonlin = (i != numberOfOutputs.size() - 1);
            layers.emplace_back(networkDimensions.at(i), networkDimensions.at(i+1), nonlin);
        }
    }

    std::vector<Node *> operator()(const std::vector<Node *> &inputVector) {
        std::vector<Node *> x = inputVector;
        for (auto &layer: layers) {
            // The output vector of a layer becomes the input vector of the other
            x = layer(x);
        }
        return x;
    }

    std::vector<Node *> parameters() override {
        // Flattens all the parameters of a layer so that they are accessible through this vector
        std::vector<Node *> params;
        for (auto &layer: layers) {
            auto layerParameters = layer.parameters();
            params.insert(params.end(), layerParameters.begin(), layerParameters.end());
        }
        return params;
    }

    std::string representation() const {
        std::string s = "Network of [";
        for (size_t i = 0; i < layers.size(); ++i) {
            s += layers.at(i).representation();
            if (i + 1 < layers.size()) s += ", ";
        }
        return s + "]";
    }
};

inline std::ostream &operator<<(std::ostream &os, const Network &m) {
    return os << m.representation();
}

#endif //NETWORK_H

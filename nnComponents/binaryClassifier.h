#ifndef BINARYCLASSIFIER_H
#define BINARYCLASSIFIER_H

#include <nnComponents/module.h>
#include <nnComponents/layer.h>

class BinaryClassifier final : public Module {
    std::vector<Layer> layers;

public:
    // For binary classification, we use sigmoid on the output layer
    BinaryClassifier(int numberOfInputs, const std::vector<int>& hiddenLayerSizes) {
        std::vector<int> networkDimensions;
        networkDimensions.reserve(hiddenLayerSizes.size() + 2);
        networkDimensions.push_back(numberOfInputs);
        networkDimensions.insert(networkDimensions.end(), hiddenLayerSizes.begin(), hiddenLayerSizes.end());
        networkDimensions.push_back(1);  // Output layer has 1 neuron for binary classification

        // Build hidden layers with ReLU
        for (size_t i = 0; i < hiddenLayerSizes.size(); ++i) {
            layers.emplace_back(networkDimensions.at(i), networkDimensions.at(i+1), Activation::RELU);
        }

        // Output layer with sigmoid activation
        layers.emplace_back(networkDimensions[hiddenLayerSizes.size()], 1, Activation::SIGMOID);
    }

    // Forward pass - returns a single output node (probability)
    Node* forward(const std::vector<Node*>& inputVector) {
        std::vector<Node*> x = inputVector;
        // This is nice because it goes in line with the idea that the output vector of one layer, serves
        // as the input vector for the next layer
        for (auto& layer : layers) {
            x = layer(x);
        }
        return x.at(0);  // Return single output for binary classification
    }

    std::vector<Node*> parameters() override {
        std::vector<Node*> params;
        for (auto& layer : layers) {
            auto layerParameters = layer.parameters();
            params.insert(params.end(), layerParameters.begin(), layerParameters.end());
        }
        return params;
    }

    std::string representation() const {
        std::string s = "BinaryClassifier of [";
        for (size_t i = 0; i < layers.size(); ++i) {
            s += layers.at(i).representation();
            if (i + 1 < layers.size()) s += ", ";
        }
        return s + "]";
    }
};

inline std::ostream& operator<<(std::ostream& os, const BinaryClassifier& m) {
    return os << m.representation();
}


#endif //BINARYCLASSIFIER_H

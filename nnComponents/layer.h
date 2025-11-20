#ifndef LAYER_H
#define LAYER_H
#include <nnComponents/module.h>
#include <nnComponents/neuron.h>

class Layer final : public Module {
    std::vector<Neuron> neurons;

public:
    Layer(int numberOfInputs, int numberOfOutputs, Activation act = Activation::RELU) {
        neurons.reserve(numberOfOutputs);
        for (int i = 0; i < numberOfOutputs; ++i) {
            neurons.emplace_back(numberOfInputs, act);
        }
    }

    std::vector<Node *> operator()(const std::vector<Node *> &x) {
        std::vector<Node *> output;
        output.reserve(neurons.size());
        for (auto &neuron: neurons) {
            output.push_back(neuron(x));
        }
        return output;
    }

    std::vector<Node *> parameters() override {
        std::vector<Node *> layerParameters;
        for (auto &neuron: neurons) {
            auto neuronParameters = neuron.parameters();
            layerParameters.insert(layerParameters.end(), neuronParameters.begin(), neuronParameters.end());
        }
        return layerParameters;
    }

    std::string representation() const {
        std::string s = "Layer of [";
        for (size_t i = 0; i < neurons.size(); ++i) {
            s += neurons.at(i).representation();
            if (i + 1 < neurons.size()) s += ", ";
        }
        return s + "]";
    }
};

inline std::ostream &operator<<(std::ostream &os, const Layer &layer) {
    return os << layer.representation();
}

#endif //LAYER_H

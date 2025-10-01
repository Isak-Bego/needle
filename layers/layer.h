#ifndef LAYER_H
#define LAYER_H
#include "neuron.h"
#include "../utils/random-generators/randomWeightGenerator.h"
#include "../utils/activation-functions/sigmoid.h"

class Layer {
    Layer *previousLayer = nullptr;
    std::vector<Neuron> neurons;

public:
    Layer() = default;

    explicit Layer(int numberOfNeurons) {
        this->neurons.reserve(numberOfNeurons); // optional optimization
        for (int n = 0; n < numberOfNeurons; ++n) {
            this->neurons.emplace_back(); // calls Neuron() constructor
        }
    }

    // In dense layers like these the number of weights in a neuron corresponds to the number of neurons in the
    // previous layer
    Layer(int numberOfNeurons, Layer *previousLayer) {
        this->previousLayer = previousLayer;
        this->neurons.reserve(numberOfNeurons);


        for (int n = 0; n < numberOfNeurons; ++n) {
            const int numberOfWeights = previousLayer->neurons.size();
            std::vector<float> weights(numberOfWeights);
            for (int w = 0; w < numberOfWeights; ++w) {
                weights[w] = static_cast<float>(generate_weight(numberOfWeights));
            }
            this->neurons.emplace_back(Neuron(weights));
        }
    }

    // Getters and setters for the private members
    std::vector<Neuron> &getNeurons() { return this->neurons; }
    void setNeurons(const std::vector<Neuron> &neurons) {this->neurons = neurons; }
    Layer *getPreviousLayer() const { return this->previousLayer; }
    void setPreviousLayer(Layer *previousLayer) { this->previousLayer = previousLayer; }

    void print() {
        for (auto &neuron: this->neurons) {
            neuron.print();
        }
    }

    // Other methods
    void forwardPass() {
        float weightedSum = 0.0;
        for (auto &neuron: this->neurons) {
            std::vector<float> weights = neuron.getWeights();
            std::vector<float> inputs;
            for (int i = 0; i < previousLayer->neurons.size(); ++i) {
                inputs.emplace_back(previousLayer->neurons.at(i).getActivation());
                weightedSum += weights.at(i) * inputs.at(i);
            }
            weightedSum = static_cast<float>(weightedSum) + neuron.getBias();
            neuron.setActivation(sigmoid(weightedSum));
            weightedSum = 0.0;
        }
    }
};


#endif //LAYER_H

#ifndef LAYER_H
#define LAYER_H
#include "neuron.h"
#include "../random-generators/randomWeightGenerator.h"
#include "../activation-functions/sigmoid.h"

class Layer {
    Layer *previousLayer = nullptr;
    std::vector<Neuron> neurons;

public:
    explicit Layer(int numberOfNeurons) {
        this->neurons.reserve(numberOfNeurons); // optional optimization
        for (int n = 0; n < numberOfNeurons; ++n) {
            this->neurons.emplace_back(); // calls Neuron() constructor
        }
    }

    //TODO: Given a previous layer we should initiazlize the Neuron with the appropriate weights. Since we are building
    // a dense neural network, meaning that each neuron in one layer is connected to each neuron in the previous layer.
    // the number of weights related to this neuron should correspond to the number of neurons in the previous layer.
    Layer(int numberOfNeurons, Layer &previousLayer) {
        this->previousLayer = &previousLayer;
        this->neurons.reserve(numberOfNeurons);
        for (int n = 0; n < numberOfNeurons; ++n) {
            const int numberOfWeights = previousLayer.neurons.size();
            std::vector<float> weights(numberOfWeights);
            for (int w = 0; w < numberOfWeights; ++w) {
                weights[w] = static_cast<float>(generate_weight(numberOfWeights));
            }
            this->neurons.emplace_back(Neuron(weights));
        }
    }

    Layer(int numberOfNeurons, Layer *previousLayer, std::vector<float> &neuronActivations) {
        this->previousLayer = previousLayer;
        this->neurons.reserve(numberOfNeurons);
        for (int n = 0; n < numberOfNeurons; ++n) {
            this->neurons.push_back(Neuron(neuronActivations.at(n)));
        }
    }

    std::vector<Neuron> &getNeurons() { return this->neurons; }

    void print() {
        for (auto &neuron: this->neurons) {
            neuron.print();
        }
    }

    Layer &back() {
        return *this->previousLayer;
    }

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

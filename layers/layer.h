#ifndef LAYER_H
#define LAYER_H
#include "neuron.h"
#include "../utils/random-generators/randomWeightGenerator.h"
#include "../utils/activation-functions/sigmoid.h"
#include "../utils/expressionNode.h"

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
            const int numberOfWeights = static_cast<int>(previousLayer->neurons.size());
            std::vector<double> weights = generateWeights(numberOfWeights);
            this->neurons.emplace_back(weights);
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
        if (this->previousLayer == nullptr) {
            return;
        }
        auto &previousNeurons = this->previousLayer->neurons;

        for (Neuron &neuron: this->neurons) {
            std::vector<Node> &weights = neuron.getWeights();
            Node *neuronActivation = &neuron.getActivation();

            for (std::size_t i = 0; i < previousNeurons.size(); ++i) {
                Node *product = weights.at(i) * previousNeurons.at(i).getActivation();
                neuronActivation = (*neuronActivation) * (*product);
            }

            neuronActivation = *neuronActivation + neuron.getBias();
            neuronActivation->apply_sigmoid();
            neuron.setActivationNode(*neuronActivation);
        }
    }

    static std::vector<double> generateWeights(const int numberOfWeights) {
        std::vector<double> weights;
        for (int w = 0; w < numberOfWeights; ++w) {
            weights.push_back(static_cast<double>(generate_weight(numberOfWeights)));
        }
        return weights;
    }

};


#endif //LAYER_H

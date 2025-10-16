#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include "utils/random-generators/randomWeightGenerator.h"
#include "utils/activation-functions/sigmoid.h"
#include "utils/activation-functions/softmax.h"

enum class LayerType {
    SIGMOID,
    SOFTMAX,
    CROSSENTROPYLOSS
};

class Layer {
    Layer *previousLayer = nullptr;
    std::vector<Neuron> neurons;
    LayerType type = LayerType::SIGMOID;
    // Store softmax outputs for backward pass
    std::vector<double> softmaxOutputs;

public:
    Layer() = default;

    explicit Layer(int numberOfNeurons, const LayerType type=LayerType::SIGMOID) {
        this->neurons.reserve(numberOfNeurons);
        this->type = type;

        for (int n = 0; n < numberOfNeurons; ++n) {
            this->neurons.emplace_back();
        }
    }

    Layer(int numberOfNeurons, Layer *previousLayer, const LayerType type=LayerType::SIGMOID) {
        this->previousLayer = previousLayer;
        this->neurons.reserve(numberOfNeurons);
        this->type = type;

        for (int n = 0; n < numberOfNeurons; ++n) {
            const int numberOfWeights = static_cast<int>(previousLayer->neurons.size());
            std::vector<double> weights = generateWeights(numberOfWeights);
            this->neurons.emplace_back(weights);
        }
    }

    // Getters and setters
    std::vector<Neuron> &getNeurons() { return this->neurons; }
    void setNeurons(const std::vector<Neuron> &neurons) {this->neurons = neurons; }
    Layer *getPreviousLayer() const { return this->previousLayer; }
    void setPreviousLayer(Layer *previousLayer) { this->previousLayer = previousLayer; }
    LayerType getType() const { return this->type; }
    const std::vector<double>& getSoftmaxOutputs() const { return this->softmaxOutputs; }

    void print() {
        std::cout<<std::endl<<"---------Layer Begin------------"<<std::endl<<std::endl;
        for (auto &neuron: this->neurons) {
            neuron.print();
        }
        std::cout<<"-----------Layer End-------------"<<std::endl;
    }

    void calculateWeightedSum(std::vector<Neuron>& previousNeurons) {
        for (Neuron &neuron: this->neurons) {
            std::vector<Node> &weights = neuron.getWeights();
            Node *neuronActivation = neuron.getActivation();

            for (std::size_t i = 0; i < previousNeurons.size(); ++i) {
                Node *product = weights.at(i) * *previousNeurons.at(i).getActivation();
                neuronActivation = (*neuronActivation) + (*product);
            }

            neuronActivation = *neuronActivation + neuron.getBias();
            neuron.setActivationNode(neuronActivation);
        }
    }

    void sigmoidForwardPass() {
        for (Neuron &neuron: this->neurons) {
            Node *neuronActivation = neuron.getActivation();
            auto* sigmoid = new Sigmoid(neuronActivation);
            neuron.setActivationNode(sigmoid);
        }
    }

    void softmaxForwardPass() {
        std::vector<double> neuronActivationValues;
        neuronActivationValues.reserve(this->neurons.size());

        for (Neuron& neuron: this->neurons) {
            neuronActivationValues.push_back(neuron.getActivation()->get_value());
        }

        std::vector<double> softmaxOutputs = Softmax::softmax(neuronActivationValues);
        for (int i = 0; i < static_cast<int>(this->neurons.size()); i++) {
            auto* softmax = new Softmax(this->neurons, softmaxOutputs, i);
            this->neurons.at(i).setActivationNode(softmax);
        }
    }

    void forwardPass() {
        if (this->previousLayer == nullptr) {
            return;
        }

        auto &previousNeurons = this->previousLayer->neurons;
        this->calculateWeightedSum(previousNeurons);

        switch (this->type) {
            case LayerType::SIGMOID:
                this->sigmoidForwardPass();
                break;
            case LayerType::SOFTMAX:
                this->softmaxForwardPass();
                break;
            default:
                break;
        }
    }

    static std::vector<double> generateWeights(const int numberOfWeights) {
        std::vector<double> weights;
        weights.reserve(numberOfWeights);
        for (int w = 0; w < numberOfWeights; ++w) {
            weights.push_back(static_cast<double>(generate_weight(numberOfWeights)));
        }
        return weights;
    }
};

#endif //LAYER_H
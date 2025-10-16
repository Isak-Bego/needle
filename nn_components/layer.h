#ifndef LAYER_H
#define LAYER_H
#include <nn_components/neuron.h>
#include <utils/random-generators/randomWeightGenerator.h>

class Layer {
    Layer *previousLayer = nullptr;
    std::vector<Neuron> neurons;

public:
    Layer() = default;

    explicit Layer(int numberOfNeurons) {
        this->neurons.reserve(numberOfNeurons);

        for (int n = 0; n < numberOfNeurons; ++n) {
            this->neurons.emplace_back();
        }
    }

    Layer(int numberOfNeurons, Layer *previousLayer) {
        this->previousLayer = previousLayer;
        this->neurons.reserve(numberOfNeurons);

        for (int n = 0; n < numberOfNeurons; ++n) {
            const int numberOfWeights = static_cast<int>(previousLayer->neurons.size());
            std::vector<double> weights = generateWeights(numberOfWeights);
            this->neurons.emplace_back(weights);
        }
    }

    virtual ~Layer() {
        delete previousLayer;
    }

    // Getters and setters
    std::vector<Neuron> &getNeurons() { return this->neurons; }
    void setNeurons(const std::vector<Neuron> &neurons) { this->neurons = neurons; }

    Layer *getPreviousLayer() const { return this->previousLayer; }
    void setPreviousLayer(Layer *previousLayer) { this->previousLayer = previousLayer; }

    virtual void print() {
        std::cout << std::endl << "---------Layer Begin------------" << std::endl << std::endl;
        for (auto &neuron: this->neurons) {
            neuron.print();
        }
        std::cout << "-----------Layer End-------------" << std::endl;
    }

    void calculateWeightedSum(std::vector<Neuron> &previousNeurons) {
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

    virtual void forwardPass() {
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

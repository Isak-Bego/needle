#ifndef NETWORK_H
#define NETWORK_H
#include <nn_components/layer.h>
#include <utils/activation-functions/softmax.h>
#include <utils/activation-functions/sigmoid.h>
#include <utils/loss-functions/crossEntropy.h>

//TODO: After you are done inspecting the functionality of Computing Partials for each neuron, make the class instance
// var private so that they abide the encapsulation principle.
class Network {
public:
    CrossEntropyLayer *errorLayer;
    std::vector<Layer *> layers;
    std::vector<std::pair<std::vector<double>, double> > trainingData;

    void wireLayers() {
        for (std::size_t i = 1; i < layers.size(); i++) {
            // We provide type safety by using at since it throws an error
            Layer *prev = this->layers.at(i - 1);
            Layer *curr = this->layers.at(i);
            curr->setPreviousLayer(prev);
        }
    }

    void initializeWeights() {
        for (std::size_t i = 1; i < layers.size(); i++) {
            Layer *prev = this->layers.at(i - 1);
            Layer *curr = this->layers.at(i);

            // Since we are creating a dense network every neuron of a layer other than the input layer is connected, to
            // every other neuron that is located in the previous layer.
            const auto previousLayerNeuronCount = prev->getNeurons().size();

            for (Neuron &neuron: curr->getNeurons()) {
                std::vector<double> weights(previousLayerNeuronCount);
                std::generate(weights.begin(), weights.end(), [&]() {
                    return generate_weight(static_cast<int>(previousLayerNeuronCount));
                });
                neuron.setWeights(weights);
            }
        }
    }

    void feedInputLayer(int inputSetNumber = 0) {
        // Assign the activation of the input layer based on the training inputs
        std::vector<Neuron> &neurons = this->layers.front()->getNeurons();
        for (std::size_t i = 0; i < neurons.size(); i++) {
            neurons.at(i).setActivation(this->trainingData.at(inputSetNumber).first[i]);
        }
        this->errorLayer->setExpectedOutput(this->trainingData.at(inputSetNumber).second);
    }

    std::vector<double> getDistinctOutputs() {
        // We first sort our outputs so that we make the counting algorithm faster
        sort(this->trainingData.begin(), this->trainingData.end(),
             [](const auto &a,
                const auto &b) {
                 return a.second < b.second;
             });

        std::vector<double> uniqueOutputs;
        double previousOutput = 0;
        bool hasStarted = false;

        // The training pair consists of: <inputVector, expectedOutput>
        for (auto &trainingPair: this->trainingData) {
            if (!hasStarted) {
                uniqueOutputs.push_back(trainingPair.second);
                previousOutput = trainingPair.second;
                hasStarted = true;
            } else if (trainingPair.second != previousOutput) {
                uniqueOutputs.push_back(trainingPair.second);
                previousOutput = trainingPair.second;
            }
        }

        return uniqueOutputs;
    }

    explicit Network(const std::vector<int> &hiddenLayerSizes) {
        for (auto layerSize: hiddenLayerSizes) {
            this->layers.push_back(new SigmoidLayer(layerSize));
        }
    }

    // Loads the training data and prepares the network for training
    void loadTrainingData(const std::vector<std::pair<std::vector<double>, double> > &trainingData) {
        this->trainingData = trainingData;
        const std::vector<double> distinctOutputs = getDistinctOutputs();
        // Create an input layer and place it at the top of the vector
        this->layers.insert(this->layers.begin(), new Layer(static_cast<int>(this->trainingData.front().first.size())));
        // This is to create the output layer
        this->layers.push_back(new SoftmaxLayer(static_cast<int>(getDistinctOutputs().size())));
        // Create the cross-entropy function that is going to calculate the error of the neural network
        this->errorLayer = new CrossEntropyLayer(1, distinctOutputs);
        this->layers.push_back(errorLayer);
        // Wires the layers and initializes all the weights
        this->wireLayers();
        this->initializeWeights();
        // We feed the input layer with the first set of outputs
        this->feedInputLayer();
    }

    void printLayers() {
        for (auto layer: layers) {
            layer->print();
        }
    }

    void forwardPass() {
        for (std::size_t j = 1; j < this->layers.size(); j++) {
            this->layers.at(j)->forwardPass();
        }
    }
};
#endif //NETWORK_H

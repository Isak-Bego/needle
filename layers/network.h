#ifndef NETWORK_H
#define NETWORK_H
#include "layer.h"

class Network {
  std::vector<Layer> layers;
  std::vector<std::vector<float>> trainingInputs;
  std::vector<float> trainingOutputs;

  void wireLayers () {
    for (std::size_t i = 1; i < layers.size(); i++) {
      // We provide type safety by using at since it throws an error
      Layer& prev = this->layers.at(i-1);
      Layer& curr = this->layers.at(i);
      curr.setPreviousLayer(&prev);
    }
  }

  void initializeWeights() {
    for (std::size_t i = 1; i < layers.size(); i++) {
      Layer& prev = this->layers.at(i-1);
      Layer& curr = this->layers.at(i);

      // Since we are creating a dense network every neuron of a layer other than the input layer is connected, to
      // every other neuron that is located in the previous layer.
      const auto previousLayerNeuronCount = prev.getNeurons().size();

      for(Neuron& neuron : curr.getNeurons()) {
        std::vector<float> weights(previousLayerNeuronCount);
        std::generate(weights.begin(), weights.end(), [&](){return generate_weight(static_cast<int>(previousLayerNeuronCount));});
        neuron.setWeights(weights);
      }
    }
  }

  void feedInputLayer(int inputSetNumber) {
    // Assign the activation of the input layer based on the training inputs
    std::vector<Neuron>& neurons = this->layers.front().getNeurons();
    for (std::size_t i = 0; i < neurons.size(); i++) {
      neurons.at(i).setActivation(this->trainingInputs.at(i)[inputSetNumber]);
    }
  }

  int getCountUniqueOutputs() {
    // We first sort our outputs so that we make the counting algorithm faster
    // TODO: We need to find a better solution for this, because we are only sorting one of the vectors.
    sort(this->trainingOutputs.begin(), this->trainingOutputs.end(),
        [](const int& a,
           const int& b) {
            return a < b;
    });

    int numberOfUniqueOutputs = 0;
    float previousOutput = 0;
    bool hasStarted = false;
    for(auto trainingOutput : this->trainingOutputs) {
      if (!hasStarted){
        ++numberOfUniqueOutputs;
        previousOutput = trainingOutput;
        hasStarted = true;
      }else if (trainingOutput != previousOutput) {
        ++numberOfUniqueOutputs;
        previousOutput = trainingOutput;
      }
    }

    return numberOfUniqueOutputs;
  }


public:
  explicit Network(const std::vector<int> &layerSizes) {
    for (auto layerSize : layerSizes) {
      this->layers.emplace_back(layerSize);
    }
  }

  // Loads the training data and prepares the network for training
  void loadTrainingData(const std::vector<std::vector<float>> &trainingInputs, const std::vector<float> &trainingOutputs){

    // Check if the sizes of the vectors are the same otherwise throw an error.
    if(trainingInputs.size() != trainingOutputs.size()) {
      throw std::logic_error("The training data is incomplete. Check the sizes of your data vectors");
    }

    // Load the data into the corresponding instance variables
    this->trainingInputs = trainingInputs;
    this->trainingOutputs = trainingOutputs;

    // Create an input layer and place it at the top of the vector
    // TODO: This implementation could be replaced with a linked list
    this->layers.emplace(this->layers.begin(), static_cast<int>(this->trainingInputs.size()));
    // We feed the input layer with the first set of outputs
    this->feedInputLayer(0);
    // This is to create the output layer
    this->layers.emplace_back(getCountUniqueOutputs());
    this->wireLayers();
    this->initializeWeights();
  }

  void printLayers() {
    for(auto layer : layers) {
      layer.print();
    }
  }

  void forwardPass() {
      for(std::size_t j = 1; j < this->layers.size(); j++) {
        this->layers.at(j).forwardPass();
      }
  }

};
#endif //NETWORK_H

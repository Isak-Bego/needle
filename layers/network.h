#ifndef NETWORK_H
#define NETWORK_H
#include "layer.h"

class Network {
  std::vector<Layer> layers;
  std::vector<std::vector<int>> trainingInputs;
  std::vector<int> trainingOutputs;

  void wireLayers () {
    for (std::size_t i = 1; i < layers.size(); i++) {
      // We provide type safety by using at since it throws an error
      Layer& prev = this->layers.at(i-1);
      Layer& curr = this->layers.at(i);
      curr.setPreviousLayer(&prev);

      //TODO: Do not forget to delete this after being done
      std::cout<<i<<"-->"<<i-1<<std::endl;
      curr.getPreviousLayer()->print();
      std::cout<<std::endl;
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

      //TODO: Do not forget to delete this after being done
      std::cout<<"Initialization of the weights was successful!"<<std::endl;
      std::cout<<i<<"-->"<<i-1<<std::endl;
      curr.getPreviousLayer()->print();
      std::cout<<std::endl;

    }
  }

  int getCountUniqueOutputs() {
    // We first sort our outputs so that we make the counting algorithm faster
    sort(this->trainingOutputs.begin(), this->trainingOutputs.end(),
        [](const int& a,
           const int& b) {
            return a < b;
    });

    //TODO: Do not forget to delete all the print statements
    std::cout<<"The sorted list of training outputs: "<<std::endl;
    for (auto trainingOutput : this->trainingOutputs) {
      std::cout<<trainingOutput << " ";
    }
    std::cout<<std::endl;

    int numberOfUniqueOutputs = 0;
    int previousOutput = 0;
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

    std::cout<<"The number of unique outputs is: "<<numberOfUniqueOutputs<<std::endl;
    return numberOfUniqueOutputs;
  }


public:
  explicit Network(const std::vector<int> &layerSizes) {
    for (auto layerSize : layerSizes) {
      this->layers.emplace_back(layerSize);
    }
  }

  // Loads the training data and prepares the network for training
  void loadTrainingData(const std::vector<std::vector<int>> &trainingInputs, const std::vector<int> &trainingOutputs){
    std::cout<<"Loading training data..."<<std::endl;

    // Check if the sizes of the vectors are the same otherwise throw an error.
    if(trainingInputs.size() != trainingOutputs.size()) {
      throw std::logic_error("The training data is incomplete. Check the sizes of your data vectors");
    }

    // Load the data into the corresponding instance variables
    this->trainingInputs = trainingInputs;
    this->trainingOutputs = trainingOutputs;

    // Create an input layer and place it at the top of the vector
    // TODO: This implementation could be replaced by a linked list
    // TODO: Don't for get to delete all the print statements
    std::cout<<"The training input size is: "<<static_cast<int>(this->trainingInputs.size())<<std::endl;
    this->layers.emplace(this->layers.begin(), static_cast<int>(this->trainingInputs.size()));
    std::cout<<"The input layer was created successfully: "<<std::endl;
    this->layers.front().print();
    std::cout<<std::endl;

    // This is to create the output layer
    this->layers.emplace_back(getCountUniqueOutputs());
    std::cout<<"The output layer was created successfully: "<<std::endl;
    this->layers.back().print();
    std::cout<<std::endl;

    this->wireLayers();
    this->initializeWeights();
  }

  void printLayers() {
    for(auto layer : layers) {
      layer.print();
    }
  }

  // void forwardPass() {
  //   for(int i = 0; i < this->training_data.size(); i++) {
  //     std::cout<<this->training_data.size();
  //     //This is odd because it does not show all the iterations, but only the first one. This has to be investigated.
  //     std::vector<Neuron> tempNeurons = this->layers.at(0).getNeurons();
  //     std::vector<int> inputValues = std::get<0>(this->training_data.at(i));
  //
  //     for (int j = 0; j < tempNeurons.size(); j++) {
  //       std::cout<<inputValues[j]<<" ";
  //       tempNeurons[j].setActivation(inputValues[j]);
  //     }
  //
  //     std::cout<<"iteration"<<i<<std::endl;
  //     this->layers.at(0).setNeurons(tempNeurons);
  //     //Forward pass form the second layer until the end
  //     for(int j = 1; j < this->layers.size(); j++) {
  //       this->layers.at(j).forwardPass();
  //     }
  //   }
  // }

};
#endif //NETWORK_H

#ifndef TRAINER_H
#define TRAINER_H
#include <iostream>

//TODO: Complete this function with the standard behaviour for training a dataset based on the examples that
// we have in the implementation of the binaryCrossEntropy and categoricalCrossEntropy. I want to include a trainer
// object in the network class or in the implementations like binaryCrossEntropy or categoricalCrossEntropy.

class Trainer {
    // Contains the network parameters
    int batchSize;
    int epochs;
    double learningRate;
    std::function<Node*()> lossFunction;
    // ... left for implementation

public:

};

#endif //TRAINER_H

#include <iostream>
#include "nn/layer.h"


// # 1
// I should have a way to reference these Neurons to one another so that I can get the information that is needed


int main() {
    std::vector<float> inputs = {1, 2, 3, 4, 5, 6, 7};

    //TODO: We can find a nicer way to bind the layers together instead of having to write the name of the previous layer
    // correctly in the constructor function
    Layer inputLayer = Layer(7, nullptr, inputs);
    Layer hiddenLayer1 = Layer(3, inputLayer);
    Layer hiddenLayer2 = Layer(3, hiddenLayer1);
    // If a previous layer exists, we iterate through all that layers neurons and through each of their weights which is
    // going to help us calculate the activation for a certain neuron
    hiddenLayer1.forwardPass();
    hiddenLayer2.forwardPass();
    hiddenLayer1.print();

    return 0;
}

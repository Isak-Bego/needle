#include <iostream>
#include "layers/layer.h"


// # 1
// I should have a way to reference these Neurons to one another so that I can get the information that is needed


int main() {
    std::vector<Neuron> inputs = {Neuron(1), Neuron(2), Neuron(3)};
    std::vector<float> output = {0, 1, 2, 3, 4, 5, 6}; // the first digit can signify which neuron should be active when the input is run
    // the second digit can be a mapping to the message we want to display
    //TODO: We can find a nicer way to bind the layers together instead of having to write the name of the previous layer
    // correctly in the constructor function
    Layer inputLayer = Layer(3);
    inputLayer.setNeurons(inputs);
    Layer hiddenLayer1 = Layer(3, &inputLayer);
    Layer hiddenLayer2 = Layer(3, &hiddenLayer1);
    // If a previous layer exists, we iterate through all that layers neurons and through each of their weights which is
    // going to help us calculate the activation for a certain neuron
    hiddenLayer1.forwardPass();
    hiddenLayer2.forwardPass();
    inputLayer.print();
    hiddenLayer1.print();
    hiddenLayer2.print();

    return 0;
}

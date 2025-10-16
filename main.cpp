#include <iostream>
#include "utils/node.h"
#include "layers/network.h"

int main() {
    std::vector<std::pair<std::vector<double>, double>> trainingData = {{{1, 2}, 1}, {{4, 5}, 2}, {{7, 8}, 1}, {{2, 4}, 1}};

    Network net = Network({2});
    net.loadTrainingData(trainingData);
    net.forwardPass();
    std::vector<Neuron> &neurons = net.layers.back().getNeurons();
    neurons.back().getActivation()->computePartials();

    net.printLayers();



    // Node a = Node(1, false);
    // Node b = Node(2, true);
    // Node c = Node(3, true);
    // Node d = Node(4, false);
    //
    // Node* e = b + c;
    // e->computePartials();
    //
    // std::cout<<"My results: "<<std::endl;
    // std::cout << a.get_gradient() << std::endl;
    // std::cout << b.get_gradient() << std::endl;
    // std::cout << c.get_gradient() << std::endl;
    // std::cout<< d.get_gradient() << std::endl;

    return 0;
}

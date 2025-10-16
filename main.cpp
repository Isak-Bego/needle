#include <iostream>
#include <utils/node.h>
#include <nn_components/network.h>

#include "utils/optimizers/sgdOptimizer.h"

int main() {
    const std::vector<std::pair<std::vector<double>, double> > trainingData = {
        {{1, 2}, 1}, {{4, 5}, 2}, {{7, 8}, 1}, {{2, 4}, 1}
    };

    auto net = Network({2});
    net.loadTrainingData(trainingData);
    net.feedInputLayer(3);
    const std::vector<Neuron> &neurons = net.getLayers().back()->getNeurons();

    net.forwardPass();
    neurons.back().getActivation()->computePartials();
    net.printLayers();

    sgdOptimizer(net);
    std::cout<<std::endl;
    std::cout <<"After the first SGD"<< std::endl;
    net.forwardPass();
    neurons.back().getActivation()->computePartials();
    net.printLayers();
    std::cout << std::endl;

    sgdOptimizer(net);
    std::cout <<"After the second SGD"<< std::endl;
    net.forwardPass();
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

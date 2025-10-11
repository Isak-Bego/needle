#include <iostream>
#include "utils/expressionNode.h"
#include "layers/network.h"

int main() {

    std::vector<std::vector<float>> trainingInput = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {2, 4, 6}};
    std::vector<float> trainingOutput = {1, 0, 1, 0};
    std::vector<std::pair<std::vector<double>, double>> trainingData = {{{1, 2, 3}, 1}, {{4, 5, 6}, 0}, {{7, 8, 9}, 1}, {{2, 4, 6}, 0}};

    Network net = Network({3});
    net.loadTrainingData(trainingData);
    net.forwardPass();
    net.printLayers();


    // Node class tests
    // Node a = Node(1, false);
    // Node b = Node(2, true);
    // Node c = Node(3, true);
    // Node d = Node(4, false);
    //
    // Node* e = b + c;
    //
    // e->computePartials();
    //
    // std::cout<<"My results: "<<std::endl;
    // std::cout << a.get_gradient() << std::endl;
    // std::cout << b.get_gradient() << std::endl;
    // std::cout << c.get_gradient() << std::endl;
    // std::cout<< d.get_gradient() << std::endl;

    return 0;
}

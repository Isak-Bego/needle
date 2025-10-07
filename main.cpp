#include <iostream>
#include "layers/layer.h"
#include "layers/network.h"




int main() {
    std::vector<std::pair<std::vector<float>, float>> trainingData = {{{1, 2, 3}, 7}, {{4, 5, 6}, 8}, {{7, 8, 9}, 9}, {{2, 4, 6}, 12}};

    Network net = Network({3, 2, 3});
    net.loadTrainingData(trainingData);
    net.forwardPass();
    net.printLayers();

    std::cout<<std::endl<<"The mean squared error is: "<<net.getMeanSquaredError();
    return 0;
}

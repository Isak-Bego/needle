#include <iostream>
#include "layers/layer.h"
#include "layers/network.h"




int main() {
    std::vector<std::pair<std::vector<float>, float>> trainingData = {{{1, 2, 3}, 1}, {{4, 5, 6}, 0}, {{7, 8, 9}, 1}, {{2, 4, 6}, 0}};

    Network net = Network({3, 2, 3});
    net.loadTrainingData(trainingData);
    net.forwardPass();
    net.printLayers();
    return 0;
}

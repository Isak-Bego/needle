#include <iostream>
#include "layers/layer.h"
#include "layers/network.h"




int main() {
    std::vector<std::vector<int>> trainingInput = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {2, 4, 6}};
    std::vector<int> trainingOutput = {1, 0, 1, 0};

    Network net = Network({3, 2, 3});
    net.loadTrainingData(trainingInput, trainingOutput);

    return 0;
}

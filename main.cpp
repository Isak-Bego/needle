#include <vector>
#include <iostream>
#include <models/multiClassClassifier.h>
#include <utils/datasets/irisDataset.h>
#include "models/binaryClassifier.h"
#include "utils/datasets/xorDataset.h"

int main() {

    // Toggle for the binaryClassifier
    // auto data = XORDataset::get_repeated(20);
    // BinaryClassifier model(data.at(0).first.size(), {8,8});
    // model.train(0.15, 1000, 1, data);
    // std::cout<<model.predict(data.at(2).first);

    auto dataset = IrisDataset::get_repeated(5);
    MultiClassClassifier model(
        IrisDataset::get_num_features(),
        {8, 8},
        IrisDataset::get_num_classes()
    );

    model.train(0.1, 500, 100, dataset);
    model.saveModel("irisclassifier.txt");

    return 0;
}
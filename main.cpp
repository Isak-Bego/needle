#include <vector>
#include <iostream>
#include <models/multiClassClassifier.h>
#include <utils/datasets/irisDataset.h>

#include "models/binaryClassifier.h"
#include "utils/datasets/xorDataset.h"


/* TODO:
 * 1. Refactor the library to a point where I have an exact idea of what is going on, where.
 * 2. Improve the convergence of the network by doing normalization of inputs
 * 3. Explain thoroughly the logic behind the binaryCrossEntropy and categoricalCrossEntropy
 */

int main() {

    // Toggle for the binaryClassifier
    // auto data = XORDataset::get_data();
    // BinaryClassifier model(data.at(0).first.size(), {8,8});
    // model.train(0.15, 1000, 1, data);
    // std::cout<<model.predict(data.at(2).first);

    auto dataset = IrisDataset::get_data();
    MultiClassClassifier model(
        IrisDataset::get_num_features(),
        {8, 8},
        IrisDataset::get_num_classes()
    );

    model.train(0.1, 2000, 100, dataset);

    std::vector<std::vector<double>> testInputs = {
        {0.22, 0.63, 0.07, 0.04},  // Should predict Setosa (0)
        {0.69, 0.42, 0.51, 0.38},  // Should predict Versicolor (1)
        {0.72, 0.50, 0.69, 0.67},  // Should predict Virginica (2)
        {0.17, 0.42, 0.07, 0.04},  // Should predict Setosa (0)
        {0.50, 0.25, 0.46, 0.38},  // Should predict Versicolor (1)
    };

    for (auto input : testInputs) {
        std::cout<<model.predict(input)<<std::endl;
    }

    return 0;
}
#include <vector>
#include <models/multiClassClassifier.h>
#include "models/binaryClassifier.h"
#include "utils/datasets/irisDataset.h"
#include "utils/datasets/mushroomDataset.h"
#include "utils/datasets/xorDataset.h"

int main() {

    auto datasetLoader = MushroomDataset("mushrooms.csv");
    auto data = datasetLoader.getData();

    MultiClassClassifier model(
        datasetLoader.getNumFeatures(),
        {12, 8},
        datasetLoader.getNumClasses()
    );

    model.train(0.15, 10, 30, data);
    model.saveModel("mushroomClassifier.txt");

    return 0;
}

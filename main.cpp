#include <vector>
#include <nnComponents/binaryClassifier.h>
#include <utils/datasets/xorDataset.h>

int main() {

    // Get dataset
    auto dataset = XORDataset::get_data();
    BinaryClassifier model(2, {8,8});
    model.train(0.2, 1000, 4, dataset);

    return 0;
}

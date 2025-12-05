#include <gtest/gtest.h>
#include <models/binaryClassifier.h>
#include <models/multiClassClassifier.h>
#include <utils/datasets/xorDataset.h>

TEST(BinaryClassifier, InitializationStructure) {
    //Given
    int inputSize = 2;
    std::vector<int> hiddenLayers = {4, 4};

    //When
    BinaryClassifier model(inputSize, hiddenLayers);
    std::vector<Node*> params = model.parameters();

    //Then
    EXPECT_GT(params.size(), 0);
}

TEST(BinaryClassifier, TrainingStepUpdatesParameters) {
    //Given
    // XOR dataset subset
    auto dataloader = XORDataset();
    auto dataset = dataloader.getData();
    BinaryClassifier model(2, {2});

    // Capture initial parameter values
    std::vector<double> initialWeights;
    for(auto* p : model.parameters()) {
        initialWeights.push_back(p->data);
    }

    //When
    model.train(0.1, 1, 1, dataset);

    //Then
    bool parametersChanged = false;
    auto newParams = model.parameters();
    for(size_t i = 0; i < newParams.size(); ++i) {
        if(std::abs(newParams.at(i)->data - initialWeights.at(i)) > 0.0) {
            parametersChanged = true;
            break;
        }
    }
    EXPECT_TRUE(parametersChanged);
}

TEST(BinaryClassifier, PredictionLogic) {
    //Given
    BinaryClassifier model(2, {2});
    std::vector<double> input = {1.0, 0.0};

    //When
    int classLabel = model.predict(input);

    //Then
    EXPECT_TRUE(classLabel == 0 || classLabel == 1);
}

TEST(MultiClassClassifier, ForwardPassShape) {
    //Given
    int inputSize = 2;
    int numClasses = 3;
    std::vector<int> hiddenLayers = {4};
    MultiClassClassifier model(inputSize, hiddenLayers, numClasses);
    std::vector<double> input = {0.5, 0.5};

    //When
    int predictedClass = model.predict(input);

    //Then
    EXPECT_GE(predictedClass, 0);
    EXPECT_LT(predictedClass, numClasses);
}


TEST(Serialization, SaveAndLoadPreservesMetadata) {
    //Given
    std::string filename = "test_model.bin";
    int inputSize = 4;
    std::vector<int> hidden = {5, 3};
    int outputClasses = 2;
    MultiClassClassifier original(inputSize, hidden, outputClasses);

    //When
    original.saveModel(filename);
    MultiClassClassifier* loaded = MultiClassClassifier::loadFromFile(filename);

    //Then
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(original.parameters().size(), loaded->parameters().size());

    // Cleanup
    delete loaded;
    std::remove(filename.c_str());
}

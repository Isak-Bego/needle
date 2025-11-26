#include <gtest/gtest.h>
#include <nnComponents/optimizers/SGD.h>
#include <nnComponents/module.h>
#include <autoGradEngine/node.h>
#include <vector>

// Mock Module for testing
class MockModule : public Module {
public:
    std::vector<Node*> params;
    MockModule() {
        params.push_back(new Node(10.0)); // Weight 1
        params.push_back(new Node(5.0));  // Weight 2
    }
    std::vector<Node*> parameters() override {
        return params;
    }
    ~MockModule() {
        for(auto* p : params) delete p;
    }
};

TEST(SGD, StepUpdatesParameters) {
    //Given
    MockModule model;
    SGD optimizer(0.1); // Learning Rate 0.1
    auto params = model.parameters();

    // Artificially set gradients
    params.at(0)->grad = 1.0;
    params.at(1)->grad = -2.0;

    double originalP0 = params.at(0)->data;
    double originalP1 = params.at(1)->data;

    //When
    optimizer.step(params);

    //Then
    EXPECT_DOUBLE_EQ(params.at(0)->data, 9.9);
    EXPECT_DOUBLE_EQ(params.at(1)->data, 5.2);
}

TEST(Module, ClearGradientsResetsToZero) {
    //Given
    MockModule model;
    auto params = model.parameters();
    params.at(0)->grad = 5.5;
    params.at(1)->grad = -3.3;

    //When
    model.clearGradients();

    //Then
    EXPECT_DOUBLE_EQ(params.at(0)->grad, 0.0);
    EXPECT_DOUBLE_EQ(params.at(1)->grad, 0.0);
}

TEST(SGD, LearningRateAccessors) {
    //Given
    SGD optimizer(0.01);

    //When
    optimizer.setLearningRate(0.05);

    //Then
    EXPECT_DOUBLE_EQ(optimizer.getLearningRate(), 0.05);
}

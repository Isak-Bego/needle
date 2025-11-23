#include <gtest/gtest.h>
#include <nnComponents/lossFunctions/binaryCrossEntropy.h>
#include <nnComponents/lossFunctions/categoricalCrossEntropy.h>
#include <vector>

TEST(LossFunctions, BCELossPerfectPrediction) {
    //Given
    // Target is 1.0, prediction is very close to 1.0
    // Loss should be near 0
    Node pred(0.9999);
    double target = 1.0;

    //When
    Node *loss = BinaryCrossEntropyLoss::compute(&pred, target);

    //Then
    EXPECT_NEAR(loss->data, 0.0, 1e-3);
    delete loss;
}

TEST(LossFunctions, BCELossBadPrediction) {
    //Given
    // Target is 1.0, prediction is 0.1
    // Loss should be high (-log(0.1) approx 2.3)
    Node pred(0.1);
    double target = 1.0;

    //When
    Node *loss = BinaryCrossEntropyLoss::compute(&pred, target);

    //Then
    EXPECT_GT(loss->data, 1.0);
    delete loss;
}

TEST(LossFunctions, CCELossSelection) {
    //Given
    // 3 Classes. Target is class index 1.
    // Case A: High probability for class 1
    std::vector<Node*> goodPreds = {new Node(0.1), new Node(0.8), new Node(0.1)};

    // Case B: Low probability for class 1
    std::vector<Node*> badPreds = {new Node(0.8), new Node(0.1), new Node(0.1)};

    //When
    Node *goodLoss = CategoricalCrossEntropyLoss::compute(goodPreds, 1);
    Node *badLoss = CategoricalCrossEntropyLoss::compute(badPreds, 1);

    //Then
    EXPECT_LT(goodLoss->data, badLoss->data);

    // Cleanup
    delete goodLoss;
    delete badLoss;
    for(auto* n : goodPreds) delete n;
    for(auto* n : badPreds) delete n;
}

TEST(LossFunctions, BCELossGradientDirection) {
    //Given
    // Target 1.0, Prediction 0.5
    // Gradient should encourage Prediction to increase (grad should be negative relative to loss minimization,
    // but in auto-diff, we check if the derivative pulls inputs towards target)
    Node pred(0.5);
    double target = 1.0;
    Node *loss = BinaryCrossEntropyLoss::compute(&pred, target);

    //When
    loss->backward();

    //Then
    // If pred increases (moves to 1.0), loss decreases.
    // So dLoss/dPred should be negative.
    EXPECT_LT(pred.grad, 0.0);
    delete loss;
}

#include <gtest/gtest.h>
#include <nnComponents/neuron.h>
#include <nnComponents/layer.h>
#include <nnComponents/activations/relu.h>
#include <nnComponents/activations/sigmoidNode.h>
#include <vector>

TEST(Activations, ReLUZeroesNegative) {
    //Given
    Node neg(-5.0);
    Node pos(5.0);

    //When
    Node *outNeg = relu(&neg);
    Node *outPos = relu(&pos);

    //Then
    EXPECT_DOUBLE_EQ(outNeg->data, 0.0);
    EXPECT_DOUBLE_EQ(outPos->data, 5.0);
    delete outNeg;
    delete outPos;
}

TEST(Activations, SigmoidRange) {
    //Given
    Node zero(0.0);

    //When
    Node *out = sigmoid(&zero);

    //Then
    EXPECT_DOUBLE_EQ(out->data, 0.5);
    delete out;
}

TEST(Neuron, ForwardPassProducesOutput) {
    //Given
    int inputSize = 3;
    Neuron n(inputSize, Activation::RELU);
    std::vector<Node *> inputs;
    inputs.push_back(new Node(1.0));
    inputs.push_back(new Node(1.0));
    inputs.push_back(new Node(1.0));

    //When
    Node *output = n(inputs);

    //Then
    EXPECT_NE(output, nullptr);
    // Output should be a valid number (actual value depends on random weights)
    EXPECT_NO_THROW(output->data);

    // Cleanup
    delete output;
    for (auto *node: inputs) delete node;
}

TEST(Neuron, ParameterCountMatchesInput) {
    //Given
    int inputSize = 5;
    Neuron n(inputSize);

    //When
    std::vector<Node *> params = n.parameters();

    //Then
    // Weights (inputSize) + Bias (1)
    EXPECT_EQ(params.size(), inputSize + 1);
}

TEST(Layer, ForwardPassDimensions) {
    //Given
    int inputSize = 2;
    int neuronsInLayer = 4;
    Layer l(inputSize, neuronsInLayer, Activation::RELU);

    std::vector<Node *> inputs;
    inputs.push_back(new Node(0.5));
    inputs.push_back(new Node(-0.5));

    //When
    std::vector<Node *> outputs = l(inputs);

    //Then
    EXPECT_EQ(outputs.size(), neuronsInLayer);

    // Cleanup
    for (auto *node: inputs) delete node;
    for (auto *node: outputs) delete node;
}

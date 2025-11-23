#include <gtest/gtest.h>
#include <autoGradEngine/node.h>
#include <nnComponents/activations/softmax.h>
#include <utils/serialization/modelSerializer.h>

TEST(Robustness, LogNodeHandlesZeroInput) {
    //Given
    // log(0) is undefined. Your implementation clamps it using epsilon.
    Node zero(0.0);
    double epsilon = 1e-7;

    //When
    Node *result = Node::logNode(&zero, epsilon);

    //Then
    // Should result in log(epsilon), not -inf or crash
    EXPECT_NEAR(result->data, std::log(epsilon), 1e-9);
    delete result;
}

TEST(Robustness, SoftmaxEmptyInput) {
    //Given
    std::vector<Node*> emptyLogits;

    //When
    std::vector<Node*> probs = softmax(emptyLogits);

    //Then
    EXPECT_TRUE(probs.empty());
}

TEST(Robustness, LoadModelInvalidFile) {
    //Given
    std::string fakePath = "non_existent_model.bin";
    std::vector<Node*> params; // Empty params

    //When
    bool success = ModelSerializer::loadWithValidation(params, fakePath);

    //Then
    EXPECT_FALSE(success);
}

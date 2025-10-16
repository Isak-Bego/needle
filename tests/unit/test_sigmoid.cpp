#include <gtest/gtest.h>
#include <cmath>

#include "utils/activation-functions/sigmoid.h"

namespace {

TEST(SigmoidStatic, SymmetryAroundOrigin) {
    constexpr double input = 1.5;
    const double positive = Sigmoid::sigmoid(input);
    const double negative = Sigmoid::sigmoid(-input);

    EXPECT_NEAR(positive, 1.0 - negative, 1e-12);
    EXPECT_GT(positive, 0.5);
    EXPECT_LT(negative, 0.5);
}

TEST(SigmoidStatic, ZeroMapsToHalf) {
    EXPECT_NEAR(Sigmoid::sigmoid(0.0), 0.5, 1e-12);
}

TEST(SigmoidDerivative, UsesOutputValue) {
    constexpr double input = 0.25;
    const double output = Sigmoid::sigmoid(input);
    const double derivative = Sigmoid::sigmoid_derivative(output);

    // Compare to analytical derivative sigma(x) * (1 - sigma(x)).
    const double expected = output * (1.0 - output);
    EXPECT_NEAR(derivative, expected, 1e-12);
}

TEST(SigmoidDerivative, GradientIsPositive) {
    constexpr double input = -4.0;
    const double output = Sigmoid::sigmoid(input);
    const double derivative = Sigmoid::sigmoid_derivative(output);

    EXPECT_GT(derivative, 0.0);
    EXPECT_LT(derivative, 0.25);
}

}  // namespace


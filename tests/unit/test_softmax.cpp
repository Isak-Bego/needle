// tests/unit/test_softmax.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "utils/activation-functions/softmax.h"
#include <vector>

// Helper for comparing doubles with relative tolerance
static inline void ExpectNearRel(double a, double b, double rel = 1e-7) {
    const double denom = std::max(std::abs(a), std::abs(b));
    EXPECT_NEAR(a, b, rel * (denom > 0 ? denom : 1.0));
}

// ---------- Softmax Forward Pass Tests ----------

TEST(SoftmaxForward, BasicProbabilities) {
    std::vector<double> logits = {1.0, 2.0, 3.0};
    std::vector<double> probs = softmax(logits);

    // Check size
    EXPECT_EQ(probs.size(), 3);

    // Check probabilities sum to 1
    double sum = 0.0;
    for (double p : probs) {
        sum += p;
    }
    ExpectNearRel(sum, 1.0);

    // Check all probabilities are positive
    for (double p : probs) {
        EXPECT_GT(p, 0.0);
        EXPECT_LT(p, 1.0);
    }

    // Largest logit should have largest probability
    EXPECT_GT(probs[2], probs[1]);
    EXPECT_GT(probs[1], probs[0]);
}

TEST(SoftmaxForward, UniformLogits) {
    std::vector<double> logits = {2.0, 2.0, 2.0, 2.0};
    std::vector<double> probs = softmax(logits);

    // All probabilities should be equal (1/4)
    for (double p : probs) {
        ExpectNearRel(p, 0.25);
    }
}

TEST(SoftmaxForward, NumericalStability) {
    // Large values that would overflow without the max subtraction trick
    std::vector<double> logits = {1000.0, 1001.0, 1002.0};
    std::vector<double> probs = softmax(logits);

    // Should still sum to 1 and not produce NaN or inf
    double sum = 0.0;
    for (double p : probs) {
        EXPECT_FALSE(std::isnan(p));
        EXPECT_FALSE(std::isinf(p));
        sum += p;
    }
    ExpectNearRel(sum, 1.0);
}

TEST(SoftmaxForward, NegativeLogits) {
    std::vector<double> logits = {-1.0, -2.0, -3.0};
    std::vector<double> probs = softmax(logits);

    // Check probabilities sum to 1
    double sum = 0.0;
    for (double p : probs) {
        sum += p;
    }
    ExpectNearRel(sum, 1.0);

    // Least negative should have highest probability
    EXPECT_GT(probs[0], probs[1]);
    EXPECT_GT(probs[1], probs[2]);
}

TEST(SoftmaxForward, EmptyInput) {
    std::vector<double> logits = {};
    std::vector<double> probs = softmax(logits);

    EXPECT_EQ(probs.size(), 0);
}

// ---------- Softmax Backward Pass Tests ----------

TEST(SoftmaxBackward, UniformGradient) {
    // If all output gradients are the same, input gradients should be zero
    std::vector<double> softmax_output = {0.25, 0.25, 0.25, 0.25};
    std::vector<double> grad_output = {1.0, 1.0, 1.0, 1.0};

    std::vector<double> grad_input = softmax_backward(softmax_output, grad_output);

    // Due to the constraint that softmax sums to 1, uniform gradients produce zero
    for (double g : grad_input) {
        ExpectNearRel(g, 0.0, 1e-10);
    }
}

TEST(SoftmaxBackward, SingleHotGradient) {
    // Typical case: gradient only on one output (like cross-entropy loss)
    std::vector<double> softmax_output = {0.1, 0.2, 0.7};
    std::vector<double> grad_output = {0.0, 0.0, 1.0}; // gradient only on 3rd output

    std::vector<double> grad_input = softmax_backward(softmax_output, grad_output);

    // Gradient should be: y_i * (delta_i3 - y_3)
    ExpectNearRel(grad_input[0], 0.1 * (0.0 - 0.7));  // -0.07
    ExpectNearRel(grad_input[1], 0.2 * (0.0 - 0.7));  // -0.14
    ExpectNearRel(grad_input[2], 0.7 * (1.0 - 0.7));  //  0.21

    // Sum of gradients should be zero (property of softmax)
    double sum = 0.0;
    for (double g : grad_input) {
        sum += g;
    }
    ExpectNearRel(sum, 0.0, 1e-10);
}

TEST(SoftmaxBackward, ConsistencyWithJacobian) {
    // Compare efficient method with Jacobian method
    std::vector<double> softmax_output = {0.2, 0.3, 0.5};
    std::vector<double> grad_output = {0.5, -0.3, 0.8};

    std::vector<double> grad_efficient = softmax_backward(softmax_output, grad_output);
    std::vector<double> grad_jacobian = softmax_backward_jacobian(softmax_output, grad_output);

    EXPECT_EQ(grad_efficient.size(), grad_jacobian.size());
    for (size_t i = 0; i < grad_efficient.size(); ++i) {
        ExpectNearRel(grad_efficient[i], grad_jacobian[i]);
    }
}

TEST(SoftmaxBackward, NumericalGradientCheck) {
    // Verify backward pass using numerical differentiation
    std::vector<double> logits = {0.5, 1.0, -0.5};
    std::vector<double> softmax_output = softmax(logits);
    std::vector<double> grad_output = {0.3, -0.2, 0.9};

    // Analytical gradient
    std::vector<double> grad_analytical = softmax_backward(softmax_output, grad_output);

    // Numerical gradient
    const double epsilon = 1e-7;
    std::vector<double> grad_numerical(logits.size());

    for (size_t i = 0; i < logits.size(); ++i) {
        // Perturb input
        std::vector<double> logits_plus = logits;
        std::vector<double> logits_minus = logits;
        logits_plus[i] += epsilon;
        logits_minus[i] -= epsilon;

        // Forward pass
        std::vector<double> output_plus = softmax(logits_plus);
        std::vector<double> output_minus = softmax(logits_minus);

        // Compute loss for both
        double loss_plus = 0.0;
        double loss_minus = 0.0;
        for (size_t j = 0; j < grad_output.size(); ++j) {
            loss_plus += grad_output[j] * output_plus[j];
            loss_minus += grad_output[j] * output_minus[j];
        }

        // Numerical gradient
        grad_numerical[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
    }

    // Compare
    for (size_t i = 0; i < grad_analytical.size(); ++i) {
        ExpectNearRel(grad_analytical[i], grad_numerical[i], 1e-5);
    }
}

TEST(SoftmaxBackward, GradientSumsToZero) {
    // Property: sum of input gradients should always be zero
    std::vector<double> softmax_output = {0.15, 0.35, 0.25, 0.25};
    std::vector<double> grad_output = {1.5, -0.8, 0.3, 2.1};

    std::vector<double> grad_input = softmax_backward(softmax_output, grad_output);

    double sum = 0.0;
    for (double g : grad_input) {
        sum += g;
    }
    ExpectNearRel(sum, 0.0, 1e-10);
}
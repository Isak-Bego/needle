// tests/test_node.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "utils/activation-functions/sigmoid.h"
#include "utils/expressionNode.h"

// Helper for near comparisons
static inline void ExpectNearRel(double a, double b, double rel = 1e-9) {
    const double denom = std::max(std::abs(a), std::abs(b));
    EXPECT_NEAR(a, b, rel * (denom > 0 ? denom : 1.0));
}

// ---------- Structure & basic ops ----------

TEST(NodeOps, AdditionBuildsBinaryNodeAndValue) {
    Node x(3.0, /*isVariable=*/true);
    Node y(4.0, /*isVariable=*/true);

    Node* sum = x + y;  // new Node owned by x
    ASSERT_NE(sum, nullptr);
    EXPECT_EQ(sum->get_operation(), '+');
    ASSERT_NE(sum->get_left(), nullptr);
    ASSERT_NE(sum->get_right(), nullptr);

    EXPECT_EQ(sum->get_left(), &x);
    EXPECT_EQ(sum->get_right(), &y);
    EXPECT_DOUBLE_EQ(sum->get_value(), 3.0 + 4.0);
}

TEST(NodeOps, MultiplicationBuildsBinaryNodeAndValue) {
    Node x(2.0, true);
    Node y(5.0, true);

    Node* prod = x * y; // new Node owned by x
    ASSERT_NE(prod, nullptr);
    EXPECT_EQ(prod->get_operation(), '*');
    ASSERT_NE(prod->get_left(), nullptr);
    ASSERT_NE(prod->get_right(), nullptr);

    EXPECT_EQ(prod->get_left(), &x);
    EXPECT_EQ(prod->get_right(), &y);
    EXPECT_DOUBLE_EQ(prod->get_value(), 2.0 * 5.0);
}

// ---------- Gradients: + and * ----------

TEST(Gradients, PlusHasUnitPartialsAtLeaves) {
    Node x(3.0, true);
    Node y(4.0, true);

    Node* z = x + y;   // z = x + y
    z->computePartials();

    // dz/dx = 1, dz/dy = 1
    ExpectNearRel(x.get_gradient(), 1.0);
    ExpectNearRel(y.get_gradient(), 1.0);
}

TEST(Gradients, MultiplyPropagatesOtherOperand) {
    Node x(3.0, true);
    Node y(4.0, true);

    Node* z = x * y;   // z = x * y
    z->computePartials();

    // dz/dx = y, dz/dy = x
    ExpectNearRel(x.get_gradient(), 4.0);
    ExpectNearRel(y.get_gradient(), 3.0);
}

TEST(Gradients, LinearComboTimesConstant) {
    Node x(1.5, true);
    Node y(-2.0, true);
    Node c(2.0, /*isVariable=*/false);

    Node* s = x + y;     // s = x + y
    Node* z = (*s) * c;  // z = (x + y) * c
    z->computePartials();

    // dz/dx = c, dz/dy = c, constant doesn't accumulate gradient
    ExpectNearRel(x.get_gradient(), 2.0);
    ExpectNearRel(y.get_gradient(), 2.0);
    ExpectNearRel(c.get_gradient(), 0.0);
}

// ---------- Sigmoid cases ----------

TEST(Gradients, SigmoidOnSingleVariableNode) {
    // Start with a single variable node (no children).
    Node x(0.7, /*isVariable=*/true);

    // Apply sigmoid in-place (your code sets isSigmoidApplied=true and mutates value).
    // After this, x.value == sigmoid(original_x).
    const double original_x = 0.7;
    x.apply_sigmoid();

    // Backprop from x as the "output".
    x.computePartials();

    // Expected: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
    // Your computePartials calls sigmoid_derivative(x.get_value()) AFTER apply,
    // so derivative should be computed using the ACTIVATED value.
    const double sx = sigmoid(original_x);
    const double expected = sigmoid_derivative(sx); // matches your implementation contract
    ExpectNearRel(x.get_gradient(), expected);
}

TEST(Gradients, SigmoidAtRootChainsThroughChildren) {
    // Build z = sigmoid(x + y)
    Node x(0.25, true);
    Node y(0.75, true);

    Node* s = x + y;     // s.value = 1.0
    s->apply_sigmoid();  // z = sigmoid(s)

    s->computePartials();

    // dz/dx = sigmoid'(1.0) * 1, dz/dy = sigmoid'(1.0) * 1
    const double s_pre = 1.0;
    const double s_act = sigmoid(s_pre);
    const double d = sigmoid_derivative(s_act); // derivative expects activated value per your code
    ExpectNearRel(x.get_gradient(), d);
    ExpectNearRel(y.get_gradient(), d);
}

// ---------- Re-entrancy / reset behavior ----------

TEST(Backprop, RecomputeDoesNotAccumulateGradients) {
    Node x(2.0, true);
    Node y(5.0, true);

    Node* z = x * y;  // z = x * y

    // First run
    z->computePartials();
    const double gx1 = x.get_gradient();
    const double gy1 = y.get_gradient();
    ExpectNearRel(gx1, 5.0);
    ExpectNearRel(gy1, 2.0);

    // Second run (should reset gradients via the pre-pass and produce same values)
    z->computePartials();
    const double gx2 = x.get_gradient();
    const double gy2 = y.get_gradient();

    ExpectNearRel(gx2, 5.0);
    ExpectNearRel(gy2, 2.0);
}

// ---------- Copy semantics (shallow copy of pointers) ----------

TEST(CopySemantics, CopyConstructorShallowPointers) {
    Node a(3.0, true);
    Node b(4.0, true);
    Node* s = a + b;      // s -> (a + b)

    Node copy(*s);        // Shallow copy
    // The copy keeps the same children pointers and metadata values
    EXPECT_EQ(copy.get_operation(), '+');
    EXPECT_EQ(copy.get_left(), s->get_left());
    EXPECT_EQ(copy.get_right(), s->get_right());
    ExpectNearRel(copy.get_value(), s->get_value());
}

#include <gtest/gtest.h>
#include <autoGradEngine/node.h>
#include <cmath>

TEST(AutoGradEngine, AdditionForward) {
    //Given
    Node a(10.0);
    Node b(20.0);

    //When
    Node *c = a + b;

    //Then
    EXPECT_DOUBLE_EQ(c->data, 30.0);
    delete c;
}

TEST(AutoGradEngine, MultiplicationForward) {
    //Given
    Node a(5.0);
    Node b(4.0);

    //When
    Node *c = a * b;

    //Then
    EXPECT_DOUBLE_EQ(c->data, 20.0);
    delete c;
}

TEST(AutoGradEngine, SimpleBackwardPass) {
    //Given
    Node x(3.0);
    Node y(4.0);
    Node *f = x * y;

    //When
    f->backward();

    //Then
    EXPECT_DOUBLE_EQ(x.grad, 4.0);
    EXPECT_DOUBLE_EQ(y.grad, 3.0);
    delete f;
}

TEST(AutoGradEngine, ComplexExpressionBackward) {
    //Given
    Node x(10.0);
    Node two(2.0);
    Node five(5.0);

    Node *step1 = x + two;
    Node *f = (*step1) * five;

    //When
    f->backward();

    //Then
    EXPECT_DOUBLE_EQ(f->data, 60.0);
    EXPECT_DOUBLE_EQ(x.grad, 5.0);

    delete step1;
    delete f;
}

TEST(AutoGradEngine, LogarithmBackward) {
    //Given
    Node x(2.0);

    //When
    Node *f = Node::logNode(&x);
    f->backward();

    //Then
    EXPECT_NEAR(f->data, std::log(2.0), 1e-7);
    EXPECT_NEAR(x.grad, 0.5, 1e-7);
    delete f;
}

TEST(AutoGradEngine, DivisionNodeNode) {
    //Given
    Node a(10.0);
    Node b(2.0);

    //When
    Node *c = a / b;
    c->backward();

    //Then
    EXPECT_DOUBLE_EQ(c->data, 5.0);
    EXPECT_DOUBLE_EQ(a.grad, 0.5); // 1/2
    EXPECT_DOUBLE_EQ(b.grad, -2.5); // -10/4
    delete c;
}

TEST(AutoGradEngine, DivisionNodeScalar) {
    //Given
    Node a(10.0);

    //When
    Node *c = a / 2.0;
    c->backward();

    //Then
    EXPECT_DOUBLE_EQ(c->data, 5.0);
    EXPECT_DOUBLE_EQ(a.grad, 0.5);
    delete c;
}

TEST(AutoGradEngine, DivisionScalarNode) {
    //Given
    Node b(2.0);

    //When
    Node *c = 10.0 / b;
    c->backward();

    //Then
    EXPECT_DOUBLE_EQ(c->data, 5.0);
    EXPECT_DOUBLE_EQ(b.grad, -2.5);
    delete c;
}

TEST(AutoGradEngine, PowerOperation) {
    //Given
    Node x(4.0);
    double exponent = 3.0;

    //When
    Node *f = x.pow(exponent);
    f->backward();

    //Then
    EXPECT_DOUBLE_EQ(f->data, 64.0);
    EXPECT_DOUBLE_EQ(x.grad, 48.0);
    delete f;
}

TEST(AutoGradEngine, ChainRuleComplex) {
    //Given
    Node x(3.0);
    Node one(1.0);

    //When
    Node *u = x + one;
    Node *f = u->pow(2.0);
    f->backward();

    //Then
    EXPECT_DOUBLE_EQ(f->data, 16.0); // 4^2
    EXPECT_DOUBLE_EQ(x.grad, 8.0);   // 2 * 4

    delete u;
    delete f;
}
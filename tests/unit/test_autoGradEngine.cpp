#include <gtest/gtest.h>
#include <autoGradEngine/node.h>

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

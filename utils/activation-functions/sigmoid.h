#ifndef SIGMOID_H
#define SIGMOID_H
#include <cmath>
#include "utils/node.h"


class Sigmoid : public Node {
public:
    explicit Sigmoid(Node *left) : Node(0.0, left, nullptr, 'f') {
        this->set_value(sigmoid(left->get_value()));
    }
    /**
    * Computes the result of the sigmoid function for a given input
    *
    * @params x The input of the function
    *
    * @returns double The result of f(x)
    */
    static double sigmoid(const double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    /**
    * Returns the value of the derivative of the sigmoid function when given the Sigmoids output.
    *
    * @params sigmoid_output The output of the sigmoid function: f(x)
    *
    * @returns The derivative of the sigmoid at x
    */
    static double sigmoid_derivative(const double sigmoid_output) {
        return sigmoid_output * (1.0 - sigmoid_output);
    }

    /**
    * Computes the partial derivative for backpropagation.
    * For sigmoid: df/dx = f(x) * (1 - f(x))
    * where f(x) is the sigmoid output we stored during forward pass
    *
    * @returns The derivative of sigmoid with respect to its input
    */
    double computeActivationPartial() override {
        return sigmoid_derivative(this->get_value());
    }
};
#endif //SIGMOID_H
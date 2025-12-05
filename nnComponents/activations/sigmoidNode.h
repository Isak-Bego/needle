#ifndef SIMOIDNODE_H
#define SIMOIDNODE_H
#include <autoGradEngine/node.h>
#include <cmath>

/**
 * @brief Takes any real valued output and squashes it in-between (0, 1).
 *
 * @param x - Sigmoid function's input
 * @return - A new node with the value of the sigmoid(x)
 */
inline Node *sigmoid(Node *x) {
    auto self = x;
    const double out_data = 1.0 / (1.0 + std::exp(-self->data));
    auto out = new Node(out_data, {self}, "sigmoid");

    // We define the logic for the backward operator
    out->backwardProp = [self, out]() {
        const double sigmoid_val = out->data;
        self->grad += sigmoid_val * (1.0 - sigmoid_val) * out->grad;
    };

    return out;
}
#endif //SIMOIDNODE_H

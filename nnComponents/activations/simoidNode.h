#ifndef SIMOIDNODE_H
#define SIMOIDNODE_H
#include <autoGradEngine/node.h>
#include <cmath>

// Sigmoid activation function as a Node method
inline Node* sigmoid(Node* x) {
    auto self = x;
    // σ(x) = 1 / (1 + e^(-x))
    double out_data = 1.0 / (1.0 + std::exp(-self->data));
    auto out = new Node(out_data, {self}, "sigmoid");

    out->_backward = [self, out]() {
        // d(σ(x))/dx = σ(x) * (1 - σ(x))
        double sigmoid_val = out->data;
        self->grad += sigmoid_val * (1.0 - sigmoid_val) * out->grad;
    };

    return out;
}
#endif //SIMOIDNODE_H

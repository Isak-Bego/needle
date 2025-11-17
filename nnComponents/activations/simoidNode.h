#ifndef SIMOIDNODE_H
#define SIMOIDNODE_H
#include <autoGradEngine/node.h>
#include <cmath>

inline Node* sigmoid(Node* x) {
    auto self = x;
    double out_data = 1.0 / (1.0 + std::exp(-self->data));
    auto out = new Node(out_data, {self}, "sigmoid");

    out->_backward = [self, out]() {
        double sigmoid_val = out->data;
        self->grad += sigmoid_val * (1.0 - sigmoid_val) * out->grad;
    };

    return out;
}
#endif //SIMOIDNODE_H

#include <autoGradEngine/node.h>

#ifndef RELU_H
#define RELU_H

/**
 * @brief It is a simple activation function that turns negative number to 0.0 and does not
 * affect the positive numbers
 *
 * @param x - the input of the ReLU (Rectified Linear Unit) function
 * @return 
 */
inline Node *relu(Node *x) {
    auto self = x;
    const auto out_data = (self->data < 0.0) ? 0.0 : self->data;
    auto out = new Node(out_data, {self}, "ReLU");

    out->_backward = [self, out]() {
        const double grad_mask = (out->data > 0.0) ? 1.0 : 0.0;
        self->grad += grad_mask * out->grad;
    };

    return out;
}

#endif //RELU_H

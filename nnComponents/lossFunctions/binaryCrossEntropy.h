#ifndef BINARYCROSSENTROPY_H
#define BINARYCROSSENTROPY_H

#include <autoGradEngine/node.h>
#include <vector>
#include <cmath>

class BinaryCrossEntropyLoss {
    static Node* log_node(Node* x, double epsilon = 1e-7) {
        auto self = x;
        double clamped = std::max(self->data, epsilon);
        double out_data = std::log(clamped);
        auto out = new Node(out_data, {self}, "log");

        out->_backward = [self, out, epsilon]() {
            double clamped = std::max(self->data, epsilon);
            self->grad += (1.0 / clamped) * out->grad;
        };

        return out;
    }

public:
    static Node* compute(Node* prediction, double target, double epsilon = 1e-7) {
        auto pred_clamped = prediction;

        auto log_pred = log_node(pred_clamped, epsilon);
        auto term1 = (*log_pred) * (-target);

        auto one_minus_pred = *((*pred_clamped) * (-1.0)) + 1.0;
        auto log_one_minus_pred = log_node(one_minus_pred, epsilon);
        auto term2 = (*log_one_minus_pred) * (-(1.0 - target));

        auto loss = (*term1) + (*term2);
        return loss;
    }
};


#endif //BINARYCROSSENTROPY_H

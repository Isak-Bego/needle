#ifndef BINARYCROSSENTROPY_H
#define BINARYCROSSENTROPY_H

#include <autoGradEngine/node.h>
#include <vector>
#include <cmath>

/**
 * @brief Computes the loss of the binary classification model.
 */
class BinaryCrossEntropyLoss {
public:
    static Node *compute(Node *prediction, double const target, double const epsilon = 1e-7) {
        const auto pred_clamped = prediction;
        const auto log_pred = Node::log_node(pred_clamped, epsilon);
        const auto term1 = (*log_pred) * (-target);

        const auto one_minus_pred = *((*pred_clamped) * (-1.0)) + 1.0;
        const auto log_one_minus_pred = Node::log_node(one_minus_pred, epsilon);
        const auto term2 = (*log_one_minus_pred) * (-(1.0 - target));

        const auto loss = (*term1) + (*term2);
        return loss;
    }
};


#endif //BINARYCROSSENTROPY_H

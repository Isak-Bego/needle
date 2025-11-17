#ifndef BINARYCROSSENTROPY_H
#define BINARYCROSSENTROPY_H

#include <autoGradEngine/node.h>
#include <vector>
#include <cmath>

class BinaryCrossEntropyLoss {
    // Natural logarithm as a Node operation
    static Node* log_node(Node* x, double epsilon = 1e-7) {
        auto self = x;
        // Clamp input to avoid log(0)
        double clamped = std::max(self->data, epsilon);
        double out_data = std::log(clamped);
        auto out = new Node(out_data, {self}, "log");

        out->_backward = [self, out, epsilon]() {
            // d(log(x))/dx = 1/x
            double clamped = std::max(self->data, epsilon);
            self->grad += (1.0 / clamped) * out->grad;
        };

        return out;
    }

public:
    // Binary cross-entropy loss for binary classification
    // L = -[y * log(p) + (1-y) * log(1-p)]
    // where p is the predicted probability and y is the true label (0 or 1)
    static Node* compute(Node* prediction, double target, double epsilon = 1e-7) {
        // Clamp prediction to avoid log(0)
        auto pred_clamped = prediction;

        // For numerical stability, we'll compute:
        // loss = -target * log(pred) - (1-target) * log(1-pred)

        // First term: -target * log(pred)
        auto log_pred = log_node(pred_clamped, epsilon);
        auto term1 = (*log_pred) * (-target);

        // Second term: -(1-target) * log(1-pred)
        auto one_minus_pred = *((*pred_clamped) * (-1.0)) + 1.0;
        auto log_one_minus_pred = log_node(one_minus_pred, epsilon);
        auto term2 = (*log_one_minus_pred) * (-(1.0 - target));

        // Total loss
        auto loss = (*term1) + (*term2);
        return loss;
    }

    // Mean loss over a batch
    static Node* compute_mean(const std::vector<Node*>& predictions,
                             const std::vector<double>& targets) {
        if (predictions.empty() || predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have same non-zero size");
        }

        Node* total_loss = new Node(0.0);
        for (size_t i = 0; i < predictions.size(); ++i) {
            auto sample_loss = compute(predictions.at(i), targets.at(i));
            total_loss = (*total_loss) + (*sample_loss);
        }

        // Divide by batch size to get mean
        auto mean_loss = (*total_loss) / static_cast<double>(predictions.size());
        return mean_loss;
    }
};


#endif //BINARYCROSSENTROPY_H

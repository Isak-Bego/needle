#ifndef CATEGORICALCROSSENTROPY_H
#define CATEGORICALCROSSENTROPY_H

#include <autoGradEngine/node.h>
#include <vector>
#include <cmath>
#include <stdexcept>

class CategoricalCrossEntropyLoss {
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
    // Categorical cross-entropy loss for multi-class classification
    // L = -sum(y_i * log(p_i))
    // where p is the predicted probability vector and y is the one-hot encoded target
    static Node* compute(const std::vector<Node*>& predictions, int target_class, 
                        double epsilon = 1e-7) {
        if (predictions.empty()) {
            throw std::invalid_argument("Predictions vector cannot be empty");
        }
        if (target_class < 0 || target_class >= static_cast<int>(predictions.size())) {
            throw std::invalid_argument("Target class out of range");
        }

        // For one-hot encoded target, only the target class contributes to loss
        // L = -log(p_target)
        auto log_prob = log_node(predictions.at(target_class), epsilon);
        auto loss = (*log_prob) * (-1.0);
        
        return loss;
    }

    // Alternative: compute with one-hot vector explicitly
    static Node* compute_with_onehot(const std::vector<Node*>& predictions,
                                     const std::vector<double>& target_onehot,
                                     double epsilon = 1e-7) {
        if (predictions.size() != target_onehot.size()) {
            throw std::invalid_argument("Predictions and target must have same size");
        }
        if (predictions.empty()) {
            throw std::invalid_argument("Predictions vector cannot be empty");
        }

        Node* total_loss = new Node(0.0);
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (target_onehot.at(i) > 0.0) {  // Only compute for non-zero targets
                auto log_prob = log_node(predictions.at(i), epsilon);
                auto term = (*log_prob) * (-target_onehot.at(i));
                total_loss = (*total_loss) + (*term);
            }
        }

        return total_loss;
    }

    // Mean loss over a batch
    static Node* compute_mean(const std::vector<std::vector<Node*>>& batch_predictions,
                             const std::vector<int>& batch_targets) {
        if (batch_predictions.empty() || batch_predictions.size() != batch_targets.size()) {
            throw std::invalid_argument("Batch predictions and targets must have same non-zero size");
        }

        Node* total_loss = new Node(0.0);
        for (size_t i = 0; i < batch_predictions.size(); ++i) {
            auto sample_loss = compute(batch_predictions.at(i), batch_targets.at(i));
            total_loss = (*total_loss) + (*sample_loss);
        }

        // Divide by batch size to get mean
        auto mean_loss = (*total_loss) / static_cast<double>(batch_predictions.size());
        return mean_loss;
    }
};

#endif //CATEGORICALCROSSENTROPY_H
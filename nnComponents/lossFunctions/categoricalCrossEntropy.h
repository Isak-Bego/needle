#ifndef CATEGORICALCROSSENTROPY_H
#define CATEGORICALCROSSENTROPY_H

#include <autoGradEngine/node.h>
#include <vector>
#include <cmath>
#include <stdexcept>

class CategoricalCrossEntropyLoss {
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
    static Node* compute(const std::vector<Node*>& predictions, int target_class, 
                        double epsilon = 1e-7) {
        if (predictions.empty()) {
            throw std::invalid_argument("Predictions vector cannot be empty");
        }
        if (target_class < 0 || target_class >= static_cast<int>(predictions.size())) {
            throw std::invalid_argument("Target class out of range");
        }

        auto log_prob = log_node(predictions.at(target_class), epsilon);
        auto loss = (*log_prob) * (-1.0);
        
        return loss;
    }
};

#endif //CATEGORICALCROSSENTROPY_H
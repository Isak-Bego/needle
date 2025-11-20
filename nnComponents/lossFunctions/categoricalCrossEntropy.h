#ifndef CATEGORICALCROSSENTROPY_H
#define CATEGORICALCROSSENTROPY_H

#include <autoGradEngine/node.h>
#include <vector>
#include <stdexcept>

/**
 * @brief Computes the value of the loss function for the multiclass classifier model
 */
class CategoricalCrossEntropyLoss {
public:
    static Node *compute(const std::vector<Node *> &predictions, int target_class,
                         const double epsilon = 1e-7) {
        if (predictions.empty()) {
            throw std::invalid_argument("Predictions vector cannot be empty");
        }
        if (target_class < 0 || target_class >= static_cast<int>(predictions.size())) {
            throw std::invalid_argument("Target class out of range");
        }

        auto log_prob = Node::log_node(predictions.at(target_class), epsilon);
        const auto loss = (*log_prob) * (-1.0);

        return loss;
    }
};

#endif //CATEGORICALCROSSENTROPY_H

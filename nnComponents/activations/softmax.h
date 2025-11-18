#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <autoGradEngine/node.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Softmax activation for multi-class classification
// Returns a vector of Nodes representing probabilities for each class
inline std::vector<Node*> softmax(const std::vector<Node*>& logits) {
    if (logits.empty()) {
        return {};
    }

    // Find max for numerical stability
    double max_val = logits.at(0)->data;
    for (size_t i = 1; i < logits.size(); ++i) {
        max_val = std::max(max_val, logits.at(i)->data);
    }

    // Compute exp(x - max) for each logit
    std::vector<Node*> exp_values;
    exp_values.reserve(logits.size());
    
    double sum_exp = 0.0;
    for (Node* logit : logits) {
        double exp_val = std::exp(logit->data - max_val);
        sum_exp += exp_val;
        exp_values.push_back(new Node(exp_val, {logit}, "exp"));
    }

    // Divide by sum to get probabilities
    std::vector<Node*> probabilities;
    probabilities.reserve(logits.size());
    
    for (size_t i = 0; i < exp_values.size(); ++i) {
        double prob = exp_values.at(i)->data / sum_exp;

        // Each probability depends on all logits, not just logits[i]. We therefore
        // record every logit as a parent so that the autograd engine includes all of
        // them in the computation graph and executes their backward functions.
        auto prob_node = new Node(prob, logits, "softmax");

        // Set up backward pass
        // Softmax gradient: prob * (1 - prob) for diagonal, -prob_i * prob_j for off-diagonal
        prob_node->_backward = [logits, i, sum_exp, exp_values, prob_node]() {
            const double prob_i = prob_node->data;

            // Gradient contribution from this output
            for (size_t j = 0; j < logits.size(); ++j) {
                const double prob_j = exp_values.at(j)->data / sum_exp;
                if (i == j) {
                    logits.at(j)->grad += prob_i * (1.0 - prob_i) * prob_node->grad;
                } else {
                    logits.at(j)->grad += -prob_i * prob_j * prob_node->grad;
                }
            }
        };

        probabilities.push_back(prob_node);
    }

    return probabilities;
}

#endif //SOFTMAX_H
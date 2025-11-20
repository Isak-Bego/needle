#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <autoGradEngine/node.h>
#include <vector>
#include <cmath>
#include <algorithm>

inline std::vector<Node*> softmax(const std::vector<Node*>& logits) {
    if (logits.empty()) {
        return {};
    }

    double max_val = logits.at(0)->data;
    for (size_t i = 1; i < logits.size(); ++i) {
        max_val = std::max(max_val, logits.at(i)->data);
    }

    std::vector<Node*> exp_values;
    exp_values.reserve(logits.size());
    
    double sum_exp = 0.0;
    for (Node* logit : logits) {
        double exp_val = std::exp(logit->data - max_val);
        sum_exp += exp_val;
        exp_values.push_back(new Node(exp_val, {logit}, "exp"));
    }

    std::vector<Node*> probabilities;
    probabilities.reserve(logits.size());
    
    for (size_t i = 0; i < exp_values.size(); ++i) {
        double prob = exp_values.at(i)->data / sum_exp;

        auto prob_node = new Node(prob, logits, "softmax");

        prob_node->_backward = [logits, i, sum_exp, exp_values, prob_node]() {
            const double prob_i = prob_node->data;

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
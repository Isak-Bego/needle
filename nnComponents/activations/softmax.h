#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <autoGradEngine/node.h>
#include <vector>
#include <cmath>

/**
 * @brief Creates the probability distribution of the logits based on their weight. The logit with the greatest value
 * is assigned a greater probability.
 *
 * @param logits - A vector of scalar values
 * @return - A vector of nodes whose data values add up to 1.
 */
inline std::vector<Node *> softmax(const std::vector<Node *> &logits) {
    if (logits.empty()) {
        return {};
    }

    // Find the maximum logit
    double maxLogit = logits.at(0)->data;
    for (size_t i = 1; i < logits.size(); ++i) {
        maxLogit = std::max(maxLogit, logits.at(i)->data);
    }

    std::vector<Node *> expValues;
    expValues.reserve(logits.size());

    // Sum all the e^(logit - maxLogit) and push into an array all the e^(logit(i) - maxLogit) for i in (1, n)
    double sum_exp = 0.0;
    for (Node *logit: logits) {
        double exp_val = std::exp(logit->data - maxLogit);
        sum_exp += exp_val;
        expValues.push_back(new Node(exp_val, {logit}, "exp"));
    }

    std::vector<Node *> probabilities;
    probabilities.reserve(logits.size());

    for (size_t i = 0; i < expValues.size(); ++i) {
        // Follows the softmax formula for finding the probabilities of each class
        double prob = expValues.at(i)->data / sum_exp;
        auto probabilityNode = new Node(prob, logits, "softmax");

        // We define the logic for the backward operator
        probabilityNode->backwardProp = [logits, i, sum_exp, expValues, probabilityNode]() {
            const double prob_i = probabilityNode->data;

            for (size_t j = 0; j < logits.size(); ++j) {
                const double prob_j = expValues.at(j)->data / sum_exp;
                if (i == j) {
                    logits.at(j)->grad += prob_i * (1.0 - prob_i) * probabilityNode->grad;
                } else {
                    logits.at(j)->grad += -prob_i * prob_j * probabilityNode->grad;
                }
            }
        };

        probabilities.push_back(probabilityNode);
    }

    return probabilities;
}

#endif //SOFTMAX_H

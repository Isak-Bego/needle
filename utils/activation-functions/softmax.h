#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "utils/expressionNode.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

class Softmax : public Node {
    std::vector<Node*> inputs;
    std::vector<double> outputs;   // y = softmax(z)
    std::vector<double> dinputs_;  // dL/dz saved here
    size_t index;

public:
    explicit Softmax(std::vector<Neuron>& in, std::vector<double>& out, int index)
        : Node(0.0, in.at(index).getActivation(), nullptr, 'f'),
          outputs(out),
          index(static_cast<size_t>(index)) {

        inputs.reserve(in.size());
        for (Neuron& n : in) {
            this->inputs.push_back(n.getActivation());
        }

        std::vector<double> scalarInputs; scalarInputs.reserve(this->inputs.size());
        for (Node* n : this->inputs) {
            scalarInputs.push_back(n->get_value());
        }

        this->set_value(this->outputs.at(this->index));
        dinputs_.assign(outputs.size(), 0.0);
        dinputs_ = softmaxInputDerivativesFromProbabilities(scalarInputs);
    }

    // Now just return the derivative wrt the index-th logit.
    double computeActivationPartial() override {
        assert(index < dinputs_.size());
        return dinputs_.at(index);
    }

    // Given softmax probabilities, compute the total derivative for each input z_j.
    // This is the column-sum of the Jacobian J = diag(p) - p p^T.
    // Result will be (near) zeros due to sum(p)=1, but computed explicitly for completeness.
    static std::vector<double> softmaxInputDerivativesFromProbabilities(const std::vector<double>& probabilities)
    {
        const size_t num_classes = probabilities.size();
        assert(num_classes > 0);

        // Build and sum Jacobian columns: d(sum_i p_i)/dz_j
        std::vector<double> d_sum_outputs_wrt_input(num_classes, 0.0);

        for (size_t j = 0; j < num_classes; ++j) {
            double column_sum = 0.0;
            for (size_t i = 0; i < num_classes; ++i) {
                const bool same_index = (i == j);
                const double jac_ij = same_index
                    ? probabilities.at(i) * (1.0 - probabilities.at(i))   // p_i * (1 - p_i)
                    : -probabilities.at(i) * probabilities.at(j);          // -p_i * p_j
                column_sum += jac_ij;
            }
            d_sum_outputs_wrt_input.at(j) = column_sum;  // will be ~0
        }
        return d_sum_outputs_wrt_input;
    }

    // Compute stable softmax probabilities from logits.
    static std::vector<double> softmax(const std::vector<double>& logits)
    {
        const size_t num_classes = logits.size();
        assert(num_classes > 0);

        // Numerical stability: subtract max logit
        const double max_logit = *std::max_element(logits.begin(), logits.end());

        std::vector<double> exp_shifted(num_classes);
        double exp_sum = 0.0;
        for (size_t i = 0; i < num_classes; ++i) {
            exp_shifted[i] = std::exp(logits.at(i) - max_logit);
            exp_sum += exp_shifted.at(i);
        }

        std::vector<double> probabilities(num_classes);
        for (size_t i = 0; i < num_classes; ++i) {
            probabilities[i] = exp_shifted[i] / exp_sum;
        }
        return probabilities;
    }
};

#endif // SOFTMAX_H

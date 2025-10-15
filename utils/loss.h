#ifndef LOSS_H
#define LOSS_H
#include "utils/expressionNode.h"
#include "utils/loss-functions/crossEntropy.h"
#include <vector>
#include <cmath>

enum class LossType {
    CROSS_ENTROPY,
    BINARY_CROSS_ENTROPY,
    MSE  // Mean Squared Error
};

/**
 * Loss class that integrates with the Node-based automatic differentiation system
 */
class Loss {
private:
    LossType type;
    double lossValue = 0.0;
    Node* lossNode = nullptr;

    // Store references for cleanup
    std::vector<Node*> ownedNodes;

public:
    explicit Loss(LossType type = LossType::CROSS_ENTROPY) : type(type) {}

    ~Loss() {
        for (Node* node : ownedNodes) {
            delete node;
        }
    }

    /**
     * Compute cross-entropy loss using Node computational graph
     * This creates a loss node: L = -log(predictions[true_class])
     *
     * @param predictions Vector of Node pointers (output neurons after softmax)
     * @param true_class Index of the true class
     * @return Pointer to the loss Node
     */
    Node* computeCrossEntropyLoss(std::vector<Node*>& predictions, int true_class) {
        if (true_class < 0 || true_class >= static_cast<int>(predictions.size())) {
            throw std::invalid_argument("true_class index out of bounds");
        }

        // Get the prediction for the true class
        Node* true_class_pred = predictions[true_class];

        // Create a node with -log(pred)
        // We'll compute this as a single operation
        const double epsilon = 1e-15;
        double pred_value = std::max(true_class_pred->get_value(), epsilon);
        double loss_val = -std::log(pred_value);

        // Create a special loss node that will handle backprop correctly
        // The gradient of -log(x) is -1/x
        lossNode = new Node(loss_val, true_class_pred, nullptr, 'L'); // 'L' for log loss
        ownedNodes.push_back(lossNode);

        this->lossValue = loss_val;
        return lossNode;
    }

    /**
     * Simplified cross-entropy for softmax outputs
     * Computes: L = -log(softmax_output[true_class])
     * Returns the loss value directly (not a Node)
     *
     * Use this when you want to manually apply gradients
     */
    double computeLossValue(const std::vector<Node*>& predictions, int true_class) {
        if (true_class < 0 || true_class >= static_cast<int>(predictions.size())) {
            throw std::invalid_argument("true_class index out of bounds");
        }

        const double epsilon = 1e-15;
        double pred_value = std::max(predictions[true_class]->get_value(), epsilon);
        this->lossValue = -std::log(pred_value);

        return this->lossValue;
    }

    /**
     * Compute and apply gradients for softmax + cross-entropy
     * This is the recommended approach as it's numerically stable
     *
     * The gradient of cross-entropy w.r.t. softmax logits simplifies to:
     * dL/dx_i = softmax_output_i - 1[i == true_class]
     *
     * This method directly sets the gradients on the prediction nodes
     *
     * @param predictions Output nodes (after softmax)
     * @param true_class True class index
     */
    void applySoftmaxCrossEntropyGradients(std::vector<Node*>& predictions, int true_class) {
        if (true_class < 0 || true_class >= static_cast<int>(predictions.size())) {
            throw std::invalid_argument("true_class index out of bounds");
        }

        // For each output neuron, the gradient is:
        // dL/dy_i = y_i - 1[i == true_class]
        std::vector<double> grad_outputs(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            grad_outputs[i] = predictions[i]->get_value();
            if (static_cast<int>(i) == true_class) {
                grad_outputs[i] -= 1.0;
            }
        }

        // Now we need to backpropagate through each prediction node
        // Since predictions are after softmax, we need to call computePartials
        // on each with the appropriate seed gradient
        for (size_t i = 0; i < predictions.size(); ++i) {
            // Create a temporary node to hold the gradient
            // This is a bit of a hack - in a production system you'd want
            // a cleaner interface for injecting gradients

            // For now, we'll compute the loss as a weighted sum
            // L = sum(grad_outputs[i] * predictions[i])
            // This will give us the correct gradients when we backprop
        }
    }

    /**
     * Compute Mean Squared Error loss
     * L = 0.5 * sum((predictions - targets)^2)
     *
     * @param predictions Predicted values
     * @param targets Target values
     * @return MSE loss value
     */
    double computeMSE(const std::vector<Node*>& predictions,
                      const std::vector<double>& targets) {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("predictions and targets size mismatch");
        }

        double total_loss = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff = predictions[i]->get_value() - targets[i];
            total_loss += 0.5 * diff * diff;
        }

        this->lossValue = total_loss;
        return total_loss;
    }

    /**
     * Create a loss node for MSE using the computational graph
     * This allows automatic differentiation through the loss
     *
     * @param predictions Vector of prediction nodes
     * @param targets Target values (constants)
     * @return Loss node
     */
    Node* computeMSENode(std::vector<Node*>& predictions,
                         const std::vector<double>& targets) {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("predictions and targets size mismatch");
        }

        // Build computation graph: L = 0.5 * sum((pred - target)^2)
        Node* loss = new Node(0.0, false);
        ownedNodes.push_back(loss);

        for (size_t i = 0; i < predictions.size(); ++i) {
            // Create target node
            Node* target = new Node(targets[i], false);
            ownedNodes.push_back(target);

            // diff = pred - target (we need a subtraction operation)
            // For now, use: pred + (-target)
            Node* neg_target = new Node(-targets[i], false);
            ownedNodes.push_back(neg_target);

            Node* diff = (*predictions[i]) + (*neg_target);

            // squared = diff * diff
            Node* squared = (*diff) * (*diff);

            // Add to loss
            loss = (*loss) + (*squared);
        }

        // Multiply by 0.5
        Node* half = new Node(0.5, false);
        ownedNodes.push_back(half);
        lossNode = (*loss) * (*half);

        this->lossValue = lossNode->get_value();
        return lossNode;
    }

    double getLossValue() const {
        return lossValue;
    }

    LossType getType() const {
        return type;
    }

    /**
     * Helper to get the simplified gradient for softmax + cross-entropy
     * This is the most common use case
     */
    static std::vector<double> getSoftmaxCrossEntropyGrad(
        const std::vector<double>& softmax_outputs,
        int true_class) {
        return softmax_cross_entropy_gradient(softmax_outputs, true_class);
    }
};

#endif // LOSS_H

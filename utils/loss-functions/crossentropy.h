#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

/**
 * Computes cross-entropy loss for a single sample
 * L = -log(y_pred[true_class])
 * 
 * @param predictions Predicted probabilities (after softmax)
 * @param true_class Index of the true class
 * @return Cross-entropy loss value
 */
double cross_entropy_loss(const std::vector<double>& predictions, int true_class) {
    if (true_class < 0 || true_class >= static_cast<int>(predictions.size())) {
        throw std::invalid_argument("true_class index out of bounds");
    }
    
    // Add small epsilon to prevent log(0)
    const double epsilon = 1e-15;
    double prob = std::max(predictions[true_class], epsilon);
    
    return -std::log(prob);
}

/**
 * Computes cross-entropy loss for a batch of samples
 * L = -(1/N) * sum_i(log(y_pred_i[true_class_i]))
 * 
 * @param predictions Vector of prediction vectors (batch_size x num_classes)
 * @param true_classes Vector of true class indices
 * @return Average cross-entropy loss
 */
double cross_entropy_loss_batch(
    const std::vector<std::vector<double>>& predictions,
    const std::vector<int>& true_classes) {
    
    if (predictions.size() != true_classes.size()) {
        throw std::invalid_argument("predictions and true_classes size mismatch");
    }
    
    if (predictions.empty()) {
        return 0.0;
    }
    
    double total_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        total_loss += cross_entropy_loss(predictions[i], true_classes[i]);
    }
    
    return total_loss / static_cast<double>(predictions.size());
}

/**
 * Computes the gradient of cross-entropy loss w.r.t. predictions
 * 
 * For cross-entropy: L = -log(y[true_class])
 * Gradient: dL/dy_i = -1/y_i if i == true_class, else 0
 * 
 * @param predictions Predicted probabilities
 * @param true_class Index of true class
 * @return Gradient vector
 */
std::vector<double> cross_entropy_gradient(
    const std::vector<double>& predictions,
    int true_class) {
    
    if (true_class < 0 || true_class >= static_cast<int>(predictions.size())) {
        throw std::invalid_argument("true_class index out of bounds");
    }
    
    std::vector<double> gradient(predictions.size(), 0.0);
    
    // Add small epsilon to prevent division by zero
    const double epsilon = 1e-15;
    double prob = std::max(predictions[true_class], epsilon);
    
    // dL/dy_i = -1/y_i for true class, 0 otherwise
    gradient[true_class] = -1.0 / prob;
    
    return gradient;
}

/**
 * Combined softmax + cross-entropy gradient (more numerically stable)
 * 
 * When softmax is followed by cross-entropy loss, the gradient simplifies to:
 * dL/dx_i = y_i - 1[i == true_class]
 * 
 * This avoids computing gradients through both softmax and cross-entropy separately,
 * which can be numerically unstable.
 * 
 * @param softmax_outputs Softmax probabilities
 * @param true_class Index of true class
 * @return Gradient w.r.t. logits (pre-softmax)
 */
std::vector<double> softmax_cross_entropy_gradient(
    const std::vector<double>& softmax_outputs,
    int true_class) {
    
    if (true_class < 0 || true_class >= static_cast<int>(softmax_outputs.size())) {
        throw std::invalid_argument("true_class index out of bounds");
    }
    
    std::vector<double> gradient(softmax_outputs.size());
    
    // The simplified gradient: y_i - 1[i == true_class]
    for (size_t i = 0; i < softmax_outputs.size(); ++i) {
        gradient[i] = softmax_outputs[i];
        if (static_cast<int>(i) == true_class) {
            gradient[i] -= 1.0;
        }
    }
    
    return gradient;
}

/**
 * Binary cross-entropy loss for binary classification
 * L = -[y*log(p) + (1-y)*log(1-p)]
 * 
 * @param prediction Predicted probability for positive class
 * @param true_label True label (0 or 1)
 * @return Binary cross-entropy loss
 */
double binary_cross_entropy(double prediction, int true_label) {
    if (true_label != 0 && true_label != 1) {
        throw std::invalid_argument("true_label must be 0 or 1");
    }
    
    // Clip prediction to prevent log(0)
    const double epsilon = 1e-15;
    prediction = std::max(std::min(prediction, 1.0 - epsilon), epsilon);
    
    if (true_label == 1) {
        return -std::log(prediction);
    } else {
        return -std::log(1.0 - prediction);
    }
}

/**
 * Gradient of binary cross-entropy
 * dL/dp = -y/p + (1-y)/(1-p)
 * 
 * @param prediction Predicted probability
 * @param true_label True label (0 or 1)
 * @return Gradient w.r.t. prediction
 */
double binary_cross_entropy_gradient(double prediction, int true_label) {
    if (true_label != 0 && true_label != 1) {
        throw std::invalid_argument("true_label must be 0 or 1");
    }
    
    const double epsilon = 1e-15;
    prediction = std::max(std::min(prediction, 1.0 - epsilon), epsilon);
    
    if (true_label == 1) {
        return -1.0 / prediction;
    } else {
        return 1.0 / (1.0 - prediction);
    }
}

#endif // CROSSENTROPY_H
#ifndef SOFTMAX_H
#define SOFTMAX_H
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

/**
 * Computes softmax activation for a vector of values
 * softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
 * 
 * Uses numerical stability trick: subtract max before exponentiating
 * to prevent overflow
 * 
 * @param x Input logits
 * @return Softmax probabilities (sum to 1.0)
 */
std::vector<double> softmax(const std::vector<double>& x) {
    if (x.empty()) {
        return std::vector<double>();
    }
    
    // Find max for numerical stability
    double max_val = *std::max_element(x.begin(), x.end());
    
    // Compute exp(x_i - max) for all elements
    std::vector<double> exp_values(x.size());
    double sum_exp = 0.0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        exp_values[i] = std::exp(x[i] - max_val);
        sum_exp += exp_values[i];
    }
    
    // Normalize to get probabilities
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = exp_values[i] / sum_exp;
    }
    
    return result;
}

/**
 * Computes the gradient of softmax with respect to inputs
 * 
 * Given:
 * - softmax_output: The output from forward pass (y = softmax(x))
 * - grad_output: Gradient from the next layer (dL/dy)
 * 
 * Computes: dL/dx using the formula:
 * dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
 * 
 * This is the efficient vectorized form of the Jacobian computation
 * 
 * @param softmax_output Output from softmax forward pass
 * @param grad_output Gradient from next layer
 * @return Gradient with respect to input (dL/dx)
 */
std::vector<double> softmax_backward(
    const std::vector<double>& softmax_output,
    const std::vector<double>& grad_output) {
    
    if (softmax_output.size() != grad_output.size()) {
        // Error: dimensions must match
        return std::vector<double>(softmax_output.size(), 0.0);
    }
    
    // Compute sum of (grad_output * softmax_output)
    double dot_product = 0.0;
    for (size_t i = 0; i < softmax_output.size(); ++i) {
        dot_product += grad_output[i] * softmax_output[i];
    }
    
    // Apply the efficient formula: dL/dx_i = y_i * (dL/dy_i - dot_product)
    std::vector<double> grad_input(softmax_output.size());
    for (size_t i = 0; i < softmax_output.size(); ++i) {
        grad_input[i] = softmax_output[i] * (grad_output[i] - dot_product);
    }
    
    return grad_input;
}

#endif // SOFTMAX_H
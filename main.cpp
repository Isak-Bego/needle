#include <iostream>
#include <iomanip>
#include <vector>

#include <autoGradEngine/node.h>
#include <nnComponents/binaryClassifier.h>
#include <utils/lossFunctions/binaryCrossEntropy.h>
#include <utils/optimizers/SGD.h>
#include <utils/datasets/xorDataset.h>

int main() {
    std::cout << "=== XOR Binary Classification with MLP ===" << std::endl;
    std::cout << std::endl;

    // Create model: 2 inputs -> [8, 8] hidden -> 1 output
    BinaryClassifier model(2, {8, 8});
    std::cout << "Model architecture:" << std::endl;
    std::cout << model << std::endl;
    std::cout << "Total parameters: " << model.parameters().size() << std::endl;
    std::cout << std::endl;

    // Create optimizer
    SGD optimizer(0.1);  // Learning rate of 0.1

    // Get dataset
    auto dataset = XORDataset::get_data();

    // Training loop
    const int epochs = 1000;
    const int print_every = 100;

    std::cout << "Training for " << epochs << " epochs..." << std::endl;
    std::cout << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        // Train on each sample
        for (const auto& sample : dataset) {
            const auto& inputs = sample.first;
            const double target = sample.second;

            // Convert inputs to Nodes
            std::vector<Node*> input_nodes;
            input_nodes.reserve(inputs.size());
            for (double val : inputs) {
                input_nodes.push_back(new Node(val));
            }

            // Forward pass
            Node* prediction = model.forward(input_nodes);

            // Compute loss
            Node* loss = BinaryCrossEntropyLoss::compute(prediction, target);
            total_loss += loss->data;

            // Backward pass
            model.clear_gradients();
            loss->backward();

            // Update weights
            std::vector<Node*> modelParams = model.parameters();
            optimizer.step(modelParams);

            // Clean up input nodes (they're not part of the model)
            for (Node* n : input_nodes) {
                delete n;
            }
        }

        // Print progress
        if ((epoch + 1) % print_every == 0) {
            double avg_loss = total_loss / dataset.size();
            std::cout << "Epoch " << std::setw(4) << (epoch + 1)
                      << " | Loss: " << std::fixed << std::setprecision(6) << avg_loss
                      << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "=== Testing on XOR dataset ===" << std::endl;
    std::cout << std::endl;

    // Test the model
    for (const auto& sample : dataset) {
        const auto& inputs = sample.first;
        const double target = sample.second;

        // Convert inputs to Nodes
        std::vector<Node*> input_nodes;
        for (double val : inputs) {
            input_nodes.push_back(new Node(val));
        }

        // Forward pass
        Node* prediction = model.forward(input_nodes);
        double pred_value = prediction->data;
        int pred_class = (pred_value >= 0.5) ? 1 : 0;

        std::cout << "Input: [" << inputs[0] << ", " << inputs[1] << "] "
                  << "| Target: " << target
                  << " | Prediction: " << std::fixed << std::setprecision(4) << pred_value
                  << " | Class: " << pred_class
                  << " | " << (pred_class == static_cast<int>(target) ? "✓ CORRECT" : "✗ WRONG")
                  << std::endl;

        // Clean up
        for (Node* n : input_nodes) {
            delete n;
        }
    }

    std::cout << std::endl;
    std::cout << "Training complete!" << std::endl;

    return 0;
}
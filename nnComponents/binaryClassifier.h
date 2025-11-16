#ifndef BINARYCLASSIFIER_H
#define BINARYCLASSIFIER_H

#include <nnComponents/network.h>
#include <nnComponents/layer.h>
#include <utils/optimizers/SGD.h>
#include <utils/lossFunctions/binaryCrossEntropy.h>
#include <iostream>
#include <iomanip>

class BinaryClassifier final : public Network {
public:
    // For binary classification, we use sigmoid on the output layer
    BinaryClassifier(int numberOfInputs, const std::vector<int>& hiddenLayerSizes) {
        std::vector<int> networkDimensions;
        networkDimensions.reserve(hiddenLayerSizes.size() + 2);
        networkDimensions.push_back(numberOfInputs);
        networkDimensions.insert(networkDimensions.end(), hiddenLayerSizes.begin(), hiddenLayerSizes.end());
        networkDimensions.push_back(1);  // Output layer has 1 neuron for binary classification

        // Build hidden layers with ReLU
        for (size_t i = 0; i < hiddenLayerSizes.size(); ++i) {
           this->layers.emplace_back(networkDimensions.at(i), networkDimensions.at(i+1), Activation::RELU);
        }

        // Output layer with sigmoid activation
        this->layers.emplace_back(networkDimensions[hiddenLayerSizes.size()], 1, Activation::SIGMOID);
    }

    // Forward pass - returns a single output node (probability)
    Node* forward(const std::vector<Node*>& inputVector) {
        std::vector<Node*> x = inputVector;
        // This is nice because it goes in line with the idea that the output vector of one layer, serves
        // as the input vector for the next layer
        for (auto& layer : this->layers) {
            x = layer(x);
        }
        return x.at(0);  // Return single output for binary classification
    }

    std::string representation() const override{
        std::string s = "BinaryClassifier of [";
        for (size_t i = 0; i < layers.size(); ++i) {
            s += layers.at(i).representation();
            if (i + 1 < layers.size()) s += ", ";
        }
        return s + "]";
    }

    void train(const double learningRate, const int epochs, const int batchSize, std::vector<std::pair<std::vector<double>, double>> dataset) override {
        const auto self = this;
        // Create optimizer
        const SGD optimizer(learningRate);
        const int print_every = epochs / 10;

        std::cout << "Training for " << epochs << " epochs..." << std::endl;
        std::cout << "The size of the dataset is: " << dataset.size() << std::endl;
        std::cout << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            int ctr = 0;

            // The vector below will be used for accumulating the param gradients
            std::vector<double> accumulatedBatchParamGradients;
            accumulatedBatchParamGradients.assign(self->parameters().size(), 0.0);


            // Train on each sample
            while (ctr < dataset.size()) {
                const auto &inputs = dataset.at(ctr).first;
                const double target = dataset.at(ctr).second;

                // Convert inputs to Nodes
                std::vector<Node *> input_nodes;
                input_nodes.reserve(inputs.size());
                for (const double val: inputs) {
                    input_nodes.push_back(new Node(val));
                }

                // Forward pass
                Node *prediction = this->forward(input_nodes);
                // Compute loss
                Node *loss = BinaryCrossEntropyLoss::compute(prediction, target);
                total_loss += loss->data;

                // Backward pass
                this->clear_gradients();
                loss->backward();

                // Add the values of the current parameter gradients to the vector with accumulated gradient
                for (size_t i = 0; i < this->parameters().size(); i++) {
                    accumulatedBatchParamGradients.at(i) += this->parameters().at(i)->grad;
                }

                if ((ctr + 1) % batchSize == 0 || ctr == dataset.size() - 1) {
                    // Update weights
                    std::vector<Node *> modelParams = this->parameters();
                    // Take the means the gradients of the parameters
                    for (size_t i = 0; i < this->parameters().size(); i++) {
                        double divisor = (ctr + 1) % batchSize != 0 ? (ctr + 1) % batchSize : batchSize;
                        accumulatedBatchParamGradients.at(i) /= divisor;
                        modelParams.at(i)->grad = accumulatedBatchParamGradients.at(i);
                    }
                    // Do Stochastic Gradient Descent
                    optimizer.step(modelParams);
                    accumulatedBatchParamGradients.assign(this->parameters().size(), 0.0);
                }

                // Clean up input nodes (they're not part of the model)
                for (const Node *n: input_nodes) {
                    delete n;
                }

                ++ctr;
            }

            ctr = 0;

            // Print progress
            if ((epoch + 1) % print_every == 0) {
                double avg_loss = total_loss / static_cast<double>(dataset.size());
                std::cout << "Epoch " << std::setw(4) << (epoch + 1)
                        << " | Loss: " << std::fixed << std::setprecision(6) << avg_loss
                        << std::endl;
            }
        }

        // Test the model
        for (const auto &sample: dataset) {
            const auto &inputs = sample.first;
            const double target = sample.second;

            // Convert inputs to Nodes
            std::vector<Node *> input_nodes;
            input_nodes.reserve(inputs.size());
            for (double val: inputs) {
                input_nodes.push_back(new Node(val));
            }

            // Forward pass
            const Node *prediction = this->forward(input_nodes);
            double pred_value = prediction->data;
            int pred_class = (pred_value >= 0.5) ? 1 : 0;

            std::cout << "Input: [" << inputs[0] << ", " << inputs[1] << "] "
                    << "| Target: " << target
                    << " | Prediction: " << std::fixed << std::setprecision(4) << pred_value
                    << " | Class: " << pred_class
                    << " | " << (pred_class == static_cast<int>(target) ? "✓ CORRECT" : "✗ WRONG")
                    << std::endl;

            // Clean up
            for (const Node *n: input_nodes) {
                delete n;
            }
        }

        std::cout << std::endl;
        std::cout << "Training complete!" << std::endl;
    }

    int predict (std::vector<double>& inputs) {
        std::vector<Node *> input_nodes;
        input_nodes.reserve(inputs.size());

        for (const double val: inputs) {
            input_nodes.push_back(new Node(val));
        }

        Node* n = forward(input_nodes);

        if (n->data >= 0.5) {
            return 1;
        }

        return 0;
    }
};


inline std::ostream& operator<<(std::ostream& os, const BinaryClassifier& m) {
    return os << m.representation();
}


#endif //BINARYCLASSIFIER_H

#ifndef MULTICLASSCLASSIFIER_H
#define MULTICLASSCLASSIFIER_H

#include <nnComponents/network.h>
#include <nnComponents/layer.h>
#include <nnComponents/optimizers/SGD.h>
#include <nnComponents/lossFunctions/categoricalCrossEntropy.h>
#include <nnComponents/activations/softmax.h>
#include <utils/serialization/modelSerializer.h>
#include <iostream>
#include <iomanip>
#include <algorithm>

class MultiClassClassifier final : public Network {
public:
    // For multi-class classification, we use softmax on the output layer
    MultiClassClassifier(const int numberOfInputs,
                        const std::vector<int>& hiddenLayerSizes,
                        const int numberOfClasses)
        : Network(getNetworkSpecs(numberOfInputs, hiddenLayerSizes, numberOfClasses)) {
    }

    /// Helper function for creating a multi-class classifier network
    static std::vector<std::pair<int, Activation>> getNetworkSpecs(
            int numberOfInputs,
            const std::vector<int>& hiddenLayerSizes,
            int numberOfClasses) {
        std::vector<std::pair<int, Activation>> networkSpecs;
        networkSpecs.emplace_back(numberOfInputs, Activation::INPUT);

        for (int hiddenLayerSize : hiddenLayerSizes) {
            networkSpecs.emplace_back(hiddenLayerSize, Activation::RELU);
        }

        // Output layer: LINEAR activation (softmax will be applied separately)
        networkSpecs.emplace_back(numberOfClasses, Activation::LINEAR);

        return networkSpecs;
    }

    std::string representation() const override {
        std::string s = "MultiClassClassifier of [";
        for (size_t i = 0; i < layers.size(); ++i) {
            s += layers.at(i).representation();
            if (i + 1 < layers.size()) s += ", ";
        }
        return s + "]";
    }

    static MultiClassClassifier* loadFromFile(const std::string& filepath) {
        try {
            // Load metadata first
            ModelMetadata metadata = ModelSerializer::loadMetadata(filepath);

            // Get number of classes from the architecture
            // Last element in hiddenLayerSizes represents output layer size
            int numClasses = metadata.hiddenLayerSizes.back();
            std::vector<int> actualHiddenLayers(
                metadata.hiddenLayerSizes.begin(),
                metadata.hiddenLayerSizes.end() - 1
            );

            // Create model with the correct architecture
            auto* model = new MultiClassClassifier(
                metadata.numInputs,
                actualHiddenLayers,
                numClasses
            );

            // Load the parameters
            std::vector<Node*> params = model->parameters();
            if (!ModelSerializer::loadWithValidation(params, filepath)) {
                delete model;
                return nullptr;
            }

            std::cout << "✓ Model loaded successfully!" << std::endl;
            std::cout << "  - Inputs: " << metadata.numInputs << std::endl;
            std::cout << "  - Hidden layers: [";
            for (size_t i = 0; i < actualHiddenLayers.size(); ++i) {
                std::cout << actualHiddenLayers.at(i);
                if (i + 1 < actualHiddenLayers.size()) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  - Output classes: " << numClasses << std::endl;
            std::cout << "  - Total parameters: " << metadata.totalParameters << std::endl;

            return model;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void train(const double learningRate, const int epochs, const int batchSize,
               std::vector<std::pair<std::vector<double>, double>>& dataset) override {
        const auto self = this;
        const SGD optimizer(learningRate);
        const int print_every = epochs / 10;

        std::cout << "Training for " << epochs << " epochs..." << std::endl;
        std::cout << "The size of the dataset is: " << dataset.size() << std::endl;
        std::cout << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            int ctr = 0;

            std::vector<double> accumulatedBatchParamGradients;
            accumulatedBatchParamGradients.assign(self->parameters().size(), 0.0);

            while (ctr < dataset.size()) {
                const auto& inputs = dataset.at(ctr).first;
                const int target_class = static_cast<int>(dataset.at(ctr).second);

                std::vector<Node*> input_nodes;
                input_nodes.reserve(inputs.size());
                for (const double val : inputs) {
                    input_nodes.push_back(new Node(val));
                }

                // Forward pass: get logits
                std::vector<Node*> logits = (*this)(input_nodes);

                // Apply softmax to get probabilities
                std::vector<Node*> probabilities = softmax(logits);

                // Compute loss
                Node* loss = CategoricalCrossEntropyLoss::compute(probabilities, target_class);
                total_loss += loss->data;

                this->clear_gradients();
                loss->backward();

                for (size_t i = 0; i < this->parameters().size(); i++) {
                    accumulatedBatchParamGradients.at(i) += this->parameters().at(i)->grad;
                }

                if ((ctr + 1) % batchSize == 0 || ctr == dataset.size() - 1) {
                    std::vector<Node*> modelParams = this->parameters();
                    for (size_t i = 0; i < this->parameters().size(); i++) {
                        double divisor = (ctr + 1) % batchSize != 0 ? (ctr + 1) % batchSize : batchSize;
                        accumulatedBatchParamGradients.at(i) /= divisor;
                        modelParams.at(i)->grad = accumulatedBatchParamGradients.at(i);
                    }

                    optimizer.step(modelParams);
                    accumulatedBatchParamGradients.assign(this->parameters().size(), 0.0);
                }

                for (const Node* n : input_nodes) {
                    delete n;
                }

                ++ctr;
            }

            ctr = 0;

            if ((epoch + 1) % print_every == 0) {
                double avg_loss = total_loss / static_cast<double>(dataset.size());
                std::cout << "Epoch " << std::setw(4) << (epoch + 1)
                        << " | Loss: " << std::fixed << std::setprecision(6) << avg_loss
                        << std::endl;
            }
        }

        // Test the model
        std::cout << "\nFinal predictions on training set:" << std::endl;
        int correct = 0;
        for (const auto& sample : dataset) {
            const auto& inputs = sample.first;
            const int target = static_cast<int>(sample.second);

            std::vector<Node*> input_nodes;
            input_nodes.reserve(inputs.size());
            for (double val : inputs) {
                input_nodes.push_back(new Node(val));
            }

            std::vector<Node*> logits = (*this)(input_nodes);
            std::vector<Node*> probabilities = softmax(logits);

            int predicted_class = 0;
            double max_prob = probabilities.at(0)->data;
            for (size_t i = 1; i < probabilities.size(); ++i) {
                if (probabilities.at(i)->data > max_prob) {
                    max_prob = probabilities.at(i)->data;
                    predicted_class = static_cast<int>(i);
                }
            }

            if (predicted_class == target) correct++;

            std::cout << "Input: [";
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << inputs[i];
                if (i + 1 < inputs.size()) std::cout << ", ";
            }
            std::cout << "] | Target: " << target
                    << " | Predicted: " << predicted_class
                    << " | Confidence: " << std::fixed << std::setprecision(4) << max_prob
                    << " | " << (predicted_class == target ? "✓ CORRECT" : "✗ WRONG")
                    << std::endl;

            for (const Node* n : input_nodes) {
                delete n;
            }
        }

        double final_accuracy = static_cast<double>(correct) / static_cast<double>(dataset.size()) * 100.0;
        std::cout << "\nFinal Training Accuracy: " << std::fixed << std::setprecision(2)
                  << final_accuracy << "%" << std::endl;
        std::cout << "Training complete!" << std::endl;
    }

    int predict(std::vector<double>& input) override {
        std::vector<Node*> input_nodes;
        input_nodes.reserve(input.size());

        for (const double val : input) {
            input_nodes.push_back(new Node(val));
        }

        std::vector<Node*> logits = (*this)(input_nodes);
        std::vector<Node*> probabilities = softmax(logits);

        // Find class with highest probability
        int predicted_class = 0;
        double max_prob = probabilities.at(0)->data;
        for (size_t i = 1; i < probabilities.size(); ++i) {
            if (probabilities.at(i)->data > max_prob) {
                max_prob = probabilities.at(i)->data;
                predicted_class = static_cast<int>(i);
            }
        }

        return predicted_class;
    }

    // Additional method to get probabilities for all classes
    std::vector<double> predict_proba(std::vector<double>& input) {
        std::vector<Node*> input_nodes;
        input_nodes.reserve(input.size());

        for (const double val : input) {
            input_nodes.push_back(new Node(val));
        }

        std::vector<Node*> logits = (*this)(input_nodes);
        std::vector<Node*> probabilities = softmax(logits);

        std::vector<double> result;
        result.reserve(probabilities.size());
        for (Node* prob : probabilities) {
            result.push_back(prob->data);
        }

        return result;
    }

    ModelMetadata getMetadata() override {
        std::vector<int> allLayerSizes;

        // Add hidden layers
        for (size_t i = 1; i < networkSpecs.size() - 1; i++) {
            allLayerSizes.push_back(networkSpecs.at(i).first);
        }

        // Add output layer (number of classes)
        allLayerSizes.push_back(networkSpecs.back().first);

        return ModelMetadata{
            this->networkSpecs.at(0).first,  // numInputs
            allLayerSizes,                    // hiddenLayerSizes + output layer
            parameters().size()               // totalParameters
        };
    }
};

inline std::ostream& operator<<(std::ostream& os, const MultiClassClassifier& m) {
    return os << m.representation();
}

#endif //MULTICLASSCLASSIFIER_H
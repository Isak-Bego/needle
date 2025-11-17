#ifndef BINARYCLASSIFIER_H
#define BINARYCLASSIFIER_H

#include <nnComponents/network.h>
#include <nnComponents/layer.h>
#include <utils/optimizers/SGD.h>
#include <utils/lossFunctions/binaryCrossEntropy.h>
#include <utils/serialization/modelSerializer.h>
#include <iostream>
#include <iomanip>

class BinaryClassifier final : public Network {
    int num_inputs;
    std::vector<int> hidden_layer_sizes;

public:
    // For binary classification, we use sigmoid on the output layer
    BinaryClassifier(int numberOfInputs, const std::vector<int>& hiddenLayerSizes)
        : num_inputs(numberOfInputs), hidden_layer_sizes(hiddenLayerSizes) {

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

    // Get model architecture info
    ModelMetadata get_metadata() {
        return ModelMetadata{num_inputs, hidden_layer_sizes, parameters().size()};
    }

    // Forward pass - returns a single output node (probability)
    Node* forward(const std::vector<Node*>& inputVector) {
        std::vector<Node*> x = inputVector;
        for (auto& layer : this->layers) {
            x = layer(x);
        }
        return x.at(0);
    }

    std::string representation() const override{
        std::string s = "BinaryClassifier of [";
        for (size_t i = 0; i < layers.size(); ++i) {
            s += layers.at(i).representation();
            if (i + 1 < layers.size()) s += ", ";
        }
        return s + "]";
    }

    // Save model with architecture metadata
    bool save_model(const std::string& filepath) {
        std::vector<Node*> params = this->parameters();
        ModelMetadata metadata = get_metadata();

        return ModelSerializer::save_with_metadata(params, metadata, filepath);

    }

    static BinaryClassifier* load_from_file(const std::string& filepath) {
        try {
            // Load metadata first
            ModelMetadata metadata = ModelSerializer::load_metadata(filepath);

            // Create model with the correct architecture
            auto* model = new BinaryClassifier(metadata.num_inputs, metadata.hidden_layer_sizes);

            // Load the parameters
            std::vector<Node*> params = model->parameters();
            if (!ModelSerializer::load_with_validation(params, filepath)) {
                delete model;
                return nullptr;
            }

            std::cout << "✓ Model loaded successfully!" << std::endl;
            std::cout << "  - Inputs: " << metadata.num_inputs << std::endl;
            std::cout << "  - Hidden layers: [";
            for (size_t i = 0; i < metadata.hidden_layer_sizes.size(); ++i) {
                std::cout << metadata.hidden_layer_sizes.at(i);
                if (i + 1 < metadata.hidden_layer_sizes.size()) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  - Total parameters: " << metadata.total_parameters << std::endl;

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
                const auto &inputs = dataset.at(ctr).first;
                const double target = dataset.at(ctr).second;

                std::vector<Node *> input_nodes;
                input_nodes.reserve(inputs.size());
                for (const double val: inputs) {
                    input_nodes.push_back(new Node(val));
                }

                Node *prediction = this->forward(input_nodes);
                Node *loss = BinaryCrossEntropyLoss::compute(prediction, target);
                total_loss += loss->data;

                this->clear_gradients();
                loss->backward();

                for (size_t i = 0; i < this->parameters().size(); i++) {
                    accumulatedBatchParamGradients.at(i) += this->parameters().at(i)->grad;
                }

                if ((ctr + 1) % batchSize == 0 || ctr == dataset.size() - 1) {
                    std::vector<Node *> modelParams = this->parameters();
                    for (size_t i = 0; i < this->parameters().size(); i++) {
                        double divisor = (ctr + 1) % batchSize != 0 ? (ctr + 1) % batchSize : batchSize;
                        accumulatedBatchParamGradients.at(i) /= divisor;
                        modelParams.at(i)->grad = accumulatedBatchParamGradients.at(i);
                    }
                    optimizer.step(modelParams);
                    accumulatedBatchParamGradients.assign(this->parameters().size(), 0.0);
                }

                for (const Node *n: input_nodes) {
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
        for (const auto &sample: dataset) {
            const auto &inputs = sample.first;
            const double target = sample.second;

            std::vector<Node *> input_nodes;
            input_nodes.reserve(inputs.size());
            for (double val: inputs) {
                input_nodes.push_back(new Node(val));
            }

            const Node *prediction = this->forward(input_nodes);
            double pred_value = prediction->data;
            int pred_class = (pred_value >= 0.5) ? 1 : 0;

            std::cout << "Input: [" << inputs[0] << ", " << inputs[1] << "] "
                    << "| Target: " << target
                    << " | Prediction: " << std::fixed << std::setprecision(4) << pred_value
                    << " | Class: " << pred_class
                    << " | " << (pred_class == static_cast<int>(target) ? "✓ CORRECT" : "✗ WRONG")
                    << std::endl;

            for (const Node *n: input_nodes) {
                delete n;
            }
        }

        std::cout << std::endl;
        std::cout << "Training complete!" << std::endl;
    }

    int predict(std::vector<double>& inputs) {
        std::vector<Node *> input_nodes;
        input_nodes.reserve(inputs.size());

        for (const double val: inputs) {
            input_nodes.push_back(new Node(val));
        }

        Node* n = forward(input_nodes);

        return (n->data >= 0.5) ? 1 : 0;
    }
};

inline std::ostream& operator<<(std::ostream& os, const BinaryClassifier& m) {
    return os << m.representation();
}

#endif //BINARYCLASSIFIER_H
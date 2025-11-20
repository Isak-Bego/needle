#ifndef BINARYCLASSIFIER_H
#define BINARYCLASSIFIER_H

#include <nnComponents/network.h>
#include <nnComponents/layer.h>
#include <nnComponents/lossFunctions/binaryCrossEntropy.h>
#include <nnComponents/trainers/trainer.h>
#include <utils/serialization/modelSerializer.h>
#include <iostream>

class BinaryClassifier final : public Network {
public:
    // For binary classification, we use sigmoid on the output layer
    BinaryClassifier(const int numberOfInputs, const std::vector<int> &hiddenLayerSizes)
        : Network(getNetworkSpecs(numberOfInputs, hiddenLayerSizes)){
    }

    /// Helper function for creating a binary classifier network
    static std::vector<std::pair<int, Activation> > getNetworkSpecs(int numberOfInputs,
                                                                    const std::vector<int> &hiddenLayerSizes) {
        std::vector<std::pair<int, Activation> > networkSpecs;
        networkSpecs.emplace_back(numberOfInputs, Activation::INPUT);
        for (int hiddenLayerSize: hiddenLayerSizes) {
            networkSpecs.emplace_back(hiddenLayerSize, Activation::RELU);
        }
        networkSpecs.emplace_back(1, Activation::SIGMOID);

        return networkSpecs;
    }

    std::string representation() const override {
        std::string s = "BinaryClassifier of [";
        for (size_t i = 0; i < layers.size(); ++i) {
            s += layers.at(i).representation();
            if (i + 1 < layers.size()) s += ", ";
        }
        return s + "]";
    }

    static BinaryClassifier *loadFromFile(const std::string &filepath) {
        try {
            // Load metadata first
            ModelMetadata metadata = ModelSerializer::loadMetadata(filepath);

            // Create model with the correct architecture
            auto *model = new BinaryClassifier(metadata.inputVectorSize, metadata.hiddenLayerSizes);

            // Load the parameters
            std::vector<Node *> params = model->parameters();
            if (!ModelSerializer::loadWithValidation(params, filepath)) {
                delete model;
                return nullptr;
            }

            std::cout << "✓ Model loaded successfully!" << std::endl;
            std::cout << "  - Input vector size: " << metadata.inputVectorSize << std::endl;
            std::cout << "  - Total parameters: " << metadata.totalParameters << std::endl;

            return model;
        } catch (const std::exception &e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void train(const double learningRate, const int epochs, const int batchSize,
               std::vector<std::pair<std::vector<double>, double> > &dataset) override {

        // Create loss function lambda
        auto loss_fn = [](const std::vector<Node*>& predictions, double target) -> Node* {
            return BinaryCrossEntropyLoss::compute(predictions.at(0), target);
        };

        // Create and configure trainer
        Trainer trainer(this, loss_fn, learningRate, epochs, batchSize);
        trainer.train(dataset);
    }

    int predict(std::vector<double> &input) override{
        std::vector<Node *> input_nodes;
        input_nodes.reserve(input.size());

        for (const double val: input) {
            input_nodes.push_back(new Node(val));
        }

        Node *n = (*this)(input_nodes).at(0);

        return (n->data >= 0.5) ? 1 : 0;
    }
};

inline std::ostream &operator<<(std::ostream &os, const BinaryClassifier &m) {
    return os << m.representation();
}

#endif //BINARYCLASSIFIER_H
#ifndef MULTICLASSCLASSIFIER_H
#define MULTICLASSCLASSIFIER_H

#include <nnComponents/network.h>
#include <nnComponents/lossFunctions/categoricalCrossEntropy.h>
#include <nnComponents/activations/softmax.h>
#include <nnComponents/trainers/trainer.h>
#include <utils/serialization/modelSerializer.h>
#include <iostream>
#include "utils/helperFunctions.h"

/**
 * A classification model that can make the distinction between n classes of objects.
 */
class MultiClassClassifier final : public Network {
    int numClasses;

public:
    MultiClassClassifier(const int numberOfInputs,
                         const std::vector<int> &hiddenLayerSizes,
                         const int numberOfClasses)
        : Network(getNetworkSpecs(numberOfInputs, hiddenLayerSizes, numberOfClasses)),
          numClasses(numberOfClasses) {
    }

    /// Helper function for creating a multi-class classifier network
    static std::vector<std::pair<int, Activation> > getNetworkSpecs(
        int numberOfInputs,
        const std::vector<int> &hiddenLayerSizes,
        int numberOfClasses) {
        std::vector<std::pair<int, Activation> > networkSpecs;
        networkSpecs.emplace_back(numberOfInputs, Activation::INPUT);

        for (int hiddenLayerSize: hiddenLayerSizes) {
            networkSpecs.emplace_back(hiddenLayerSize, Activation::RELU);
        }

        // Output layer: LINEAR activation (softmax will be applied separately)
        // The last layer will contain the logits that we will turn into a probability distribution using the softmax.
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

    /**
     *
     * @param filepath - location of the .bin file that holds the metadata + parameters of a saved model
     * @return
     */
    static MultiClassClassifier *loadFromFile(const std::string &filepath) {
        try {
            // Load metadata first
            ModelMetadata metadata = ModelSerializer::loadMetadata(filepath);

            // Get number of classes from the architecture
            // Last element in hiddenLayerSizes represents output layer size
            int const numClasses = metadata.hiddenLayerSizes.back();
            std::vector<int> const actualHiddenLayers(
                metadata.hiddenLayerSizes.begin(),
                metadata.hiddenLayerSizes.end() - 1
            );

            // Create model with the correct architecture
            auto *model = new MultiClassClassifier(
                metadata.inputVectorSize,
                actualHiddenLayers,
                numClasses
            );

            // Load the parameters
            std::vector<Node *> params = model->parameters();
            if (!ModelSerializer::loadWithValidation(params, filepath)) {
                delete model;
                return nullptr;
            }

            std::cout << "✓ Model loaded successfully!" << std::endl;
            std::cout << "  - Input vector size: " << metadata.inputVectorSize << std::endl;
            std::cout << "  - Output classes: " << numClasses << std::endl;
            std::cout << "  - Total parameters: " << metadata.totalParameters << std::endl;

            return model;
        } catch (const std::exception &e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return nullptr;
        }
    }

    /**
     *
     * @param learningRate - the rate at which we will be updating the parameters of the network
     * @param epochs - the number of iterations through the training data
     * @param batchSize - the number of samples considered before making a parameter update
     * @param dataset - the dataset that we are going to be using to train the network
     */
    void train(const double learningRate, const int epochs, const int batchSize,
               std::vector<std::pair<std::vector<double>, double> > &dataset) override {
        // Create loss function lambda that handles softmax + cross-entropy
        auto loss_fn = [this](const std::vector<Node *> &logits, const double target) -> Node *{
            // Apply softmax to logits
            const std::vector<Node *> probabilities = softmax(logits);

            // Compute categorical cross-entropy loss
            Node *loss = CategoricalCrossEntropyLoss::compute(probabilities, static_cast<int>(target));

            return loss;
        };

        // Create and configure trainer
        Trainer trainer(this, loss_fn, learningRate, epochs, batchSize);
        trainer.train(dataset);
    }

    int predict(std::vector<double> &input) override {
        const auto inputNodes = helper::createInputNodes(input);

        const std::vector<Node *> logits = (*this)(inputNodes);
        std::vector<Node *> probabilities = softmax(logits);

        // Find class with the highest probability
        int predictedClass = 0;
        double maxProbability = probabilities.at(0)->data;
        for (size_t i = 1; i < probabilities.size(); ++i) {
            if (probabilities.at(i)->data > maxProbability) {
                maxProbability = probabilities.at(i)->data;
                predictedClass = static_cast<int>(i);
            }
        }

        return predictedClass;
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
            this->networkSpecs.at(0).first,
            allLayerSizes,
            parameters().size()
        };
    }
};

inline std::ostream &operator<<(std::ostream &os, const MultiClassClassifier &m) {
    return os << m.representation();
}

#endif //MULTICLASSCLASSIFIER_H

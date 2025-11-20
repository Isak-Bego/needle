#ifndef BINARYCLASSIFIER_H
#define BINARYCLASSIFIER_H

#include <nnComponents/network.h>
#include <nnComponents/layer.h>
#include <nnComponents/lossFunctions/binaryCrossEntropy.h>
#include <nnComponents/trainers/trainer.h>
#include <utils/serialization/modelSerializer.h>
#include <iostream>

/**
 * A classification model that can make the distinction between two classes of objects.
 */
class BinaryClassifier final : public Network {
public:
    BinaryClassifier(const int numberOfInputs, const std::vector<int> &hiddenLayerSizes)
        : Network(getNetworkSpecs(numberOfInputs, hiddenLayerSizes)) {
    }

    /// Helper function for creating a binary classifier network
    static std::vector<std::pair<int, Activation> > getNetworkSpecs(int numberOfInputs,
                                                                    const std::vector<int> &hiddenLayerSizes) {
        std::vector<std::pair<int, Activation> > networkSpecs;
        networkSpecs.emplace_back(numberOfInputs, Activation::INPUT);
        for (int hiddenLayerSize: hiddenLayerSizes) {
            networkSpecs.emplace_back(hiddenLayerSize, Activation::RELU);
        }
        // For binary classification, we use sigmoid on the output layer, so that we get a number in the range
        // (0, 1)
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

    /**
     *
     * @param filepath - location of the .bin file that holds the metadata + parameters of a saved model
     * @return
     */
    static BinaryClassifier *loadFromFile(const std::string &filepath) {
        try {
            // Load metadata first
            const ModelMetadata metadata = ModelSerializer::loadMetadata(filepath);

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

    /**
     *
     * @param learningRate - the rate at which we will be updating the parameters of the network
     * @param epochs - the number of iterations through the training data
     * @param batchSize - the number of samples considered before making a parameter update
     * @param dataset - the dataset that we are going to be using to train the network
     */
    void train(const double learningRate, const int epochs, const int batchSize,
               std::vector<std::pair<std::vector<double>, double> > &dataset) override {
        // Create loss function lambda so that we pass it to the trainer
        auto loss_fn = [](const std::vector<Node *> &predictions, const double target) -> Node *{
            return BinaryCrossEntropyLoss::compute(predictions.at(0), target);
        };

        // Create the trainer object and then call the train method to start training the network
        Trainer trainer(this, loss_fn, learningRate, epochs, batchSize);
        trainer.train(dataset);
    }

    /**
     * @brief It is used to make inferences with a trained model.
     *
     * @param input - The input vector
     * @return - The number of the class that was predicted by the model
     */
    int predict(std::vector<double> &input) override {
        auto inputNodes = helper::createInputNodes(input);

        for (const double val: input) {
            inputNodes.push_back(new Node(val));
        }

        const Node *n = (*this)(inputNodes).at(0);
        auto prediction = n->data;

        helper::deleteInputNodes(inputNodes);

        return (prediction >= 0.5) ? 1 : 0;
    }
};

inline std::ostream &operator<<(std::ostream &os, const BinaryClassifier &m) {
    return os << m.representation();
}

#endif //BINARYCLASSIFIER_H

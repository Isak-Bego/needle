#ifndef TRAINER_H
#define TRAINER_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <functional>
#include <autoGradEngine/node.h>
#include <nnComponents/optimizers/SGD.h>
#include "utils/helperFunctions.h"

class Trainer {
    Network* network;
    std::function<Node*(const std::vector<Node*>&, double)> lossFunction;
    SGD optimizer;
    int epochs;
    int batchSize;
    int printEvery;
    bool verbose;

public:
    /**
     * @brief Construct a Trainer for a neural network
     *
     * @param net - pointer to the network to train
     * @param loss_fn - loss function that computes loss given predictions and target
     * @param learningRate - learning rate for SGD optimizer
     * @param num_epochs - number of training epochs
     * @param batch_size - batch size for gradient accumulation
     * @param print_frequency - print loss every N epochs (default: epochs/10)
     */
    Trainer(Network* net,
            const std::function<Node*(const std::vector<Node*>&, double)>& loss_fn,
            double learningRate = 0.01,
            int num_epochs = 100,
            int batch_size = 32,
            int print_frequency = -1)
        : network(net),
          lossFunction(loss_fn),
          optimizer(learningRate),
          epochs(num_epochs),
          batchSize(batch_size),
          verbose(true) {

        // Default: print 10 times during training
        printEvery = (print_frequency <= 0) ? std::max(1, num_epochs / 10) : print_frequency;
    }

    /**
     * @brief Set the learning rate
     */
    void setLearningRate(double lr) {
        optimizer.set_learning_rate(lr);
    }

    /**
     * @brief Get the current learning rate
     */
    double getLearningRate() const {
        return optimizer.get_learning_rate();
    }

    /**
     * @brief Set the number of epochs
     */
    void setEpochs(int num_epochs) {
        epochs = num_epochs;
        printEvery = std::max(1, num_epochs / 10);
    }

    /**
     * @brief Set batch size
     */
    void setBatchSize(int size) {
        batchSize = std::max(1, size);
    }

    /**
     * @brief Enable or disable verbose output
     */
    void setVerbose(bool enable) {
        verbose = enable;
    }

    /**
     * @brief Train the network on the provided dataset
     *
     * @param dataset - vector of (input_vector, target_label) pairs
     * @return average loss after training
     */
    double train(const std::vector<std::pair<std::vector<double>, double>>& dataset) {
        if (dataset.empty()) {
            throw std::invalid_argument("Dataset cannot be empty");
        }

        if (!network) {
            throw std::invalid_argument("Network pointer is null");
        }

        if (verbose) {
            std::cout << "Training for " << epochs << " epochs..." << std::endl;
            std::cout << "Dataset size: " << dataset.size() << std::endl;
            std::cout << "Batch size: " << batchSize << std::endl;
            std::cout << std::endl;
        }

        double totalLoss = 0.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epochLoss = 0.0;
            int sampleCount = 0;

            // Accumulator for gradient averaging within a batch
            std::vector<double> accumulatedGradients(network->parameters().size(), 0.0);

            // Iterate through dataset
            for (const auto& sample : dataset) {
                const auto& inputs = sample.first;
                const double target = sample.second;

                auto inputNodes = helper::createInputNodes(inputs);

                // Forward pass
                std::vector<Node*> predictions = (*network)(inputNodes);

                // Compute loss
                Node* loss = lossFunction(predictions, target);
                epochLoss += loss->data;

                // Backward pass
                network->clear_gradients();
                loss->backward();

                // Accumulate gradients
                auto params = network->parameters();
                for (size_t i = 0; i < params.size(); ++i) {
                    accumulatedGradients.at(i) += params.at(i)->grad;
                }

                ++sampleCount;

                // Update parameters when batch is full or at end of dataset
                if (sampleCount % batchSize == 0 || sampleCount == static_cast<int>(dataset.size())) {
                    // Average gradients over batch
                    int batchSizeUsed = (sampleCount % batchSize == 0) ? batchSize : (sampleCount % batchSize);
                    for (size_t i = 0; i < params.size(); ++i) {
                        params.at(i)->grad = accumulatedGradients.at(i) / batchSizeUsed;
                    }

                    // Optimizer step
                    optimizer.step(params);

                    // Reset accumulator
                    accumulatedGradients.assign(params.size(), 0.0);
                }

                helper::deleteInputNodes(inputNodes);
            }

            totalLoss = epochLoss;

            // Print progress
            if (verbose && (epoch + 1) % printEvery == 0) {
                double avgLoss = epochLoss / static_cast<double>(dataset.size());
                std::cout << "Epoch " << std::setw(4) << (epoch + 1)
                         << " | Loss: " << std::fixed << std::setprecision(6) << avgLoss
                         << std::endl;
            }
        }

        if (verbose) {
            std::cout << "\nTraining complete!" << std::endl;
        }

        return totalLoss / static_cast<double>(dataset.size());
    }
};

#endif //TRAINER_H
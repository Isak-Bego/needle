#ifndef TRAINER_H
#define TRAINER_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <functional>
#include <autoGradEngine/node.h>
#include <nnComponents/optimizers/SGD.h>
#include "utils/helperFunctions.h"

using DatasetFormat = std::vector<std::pair<std::vector<double>, double> >;

class Trainer {
    Network *network;
    std::function<Node*(const std::vector<Node *> &, double)> lossFunction;
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
     * @param epochsNum - number of training epochs
     * @param batchSize - batch size for gradient accumulation
     * @param printFrequency - print loss every N epochs (default: epochs/10)
     */
    Trainer(Network *net,
            const std::function<Node*(const std::vector<Node *> &, double)> &loss_fn,
            const double learningRate = 0.01,
            const int epochsNum = 100,
            const int batchSize = 32,
            const int printFrequency = -1)
        : network(net),
          lossFunction(loss_fn),
          optimizer(learningRate),
          epochs(epochsNum),
          batchSize(batchSize),
          verbose(true) {
        // Default: print 10 times during training
        printEvery = std::max(1, epochsNum / 100);
    }

    /**
     * @brief Set the learning rate
     */
    void setLearningRate(const double lr) {
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
    void setEpochs(const int num_epochs) {
        epochs = num_epochs;
        printEvery = std::max(1, num_epochs / 10);
    }

    /**
     * @brief Set batch size
     */
    void setBatchSize(const int size) {
        batchSize = std::max(1, size);
    }

    /**
     * @brief Enable or disable verbose output
     */
    void setVerbose(const bool enable) {
        verbose = enable;
    }

    double computeAccuracy(const DatasetFormat &subset) {

        if (subset.empty()) return 0.0;

        int correct = 0;

        for (auto sample: subset) {
            auto &inputs = sample.first;
            double target = sample.second;

            auto predictedClass = network->predict(inputs);

            if (predictedClass == static_cast<int>(target)) {
                correct++;
            }
        }

        return static_cast<double>(correct) / static_cast<double>(subset.size());
    }

    /*
     * Splits the data into training data, validation data and test data.
     ***/
    static std::tuple<DatasetFormat, DatasetFormat, DatasetFormat> splitData(const DatasetFormat &dataset) {
        DatasetFormat datasetCopy = dataset;
        int trainingDatasetSize, validationDatasetSize;
        int datasetSize = static_cast<int>(dataset.size());
        // We shuffle the dataset to make sure everything is random
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(datasetCopy.begin(), datasetCopy.end(), g);

        if (datasetSize < 100) {
            //Ratios 60%-20%-20%
            trainingDatasetSize = static_cast<int>(datasetSize * 0.6);
            validationDatasetSize = static_cast<int>(datasetSize * 0.2);
        } else if (datasetSize < 100000) {
            //Ratios 70%-15%-15%
            trainingDatasetSize = static_cast<int>(datasetSize * 0.70);
            validationDatasetSize = static_cast<int>(datasetSize * 0.15);
        } else {
            //Ratios 98%-1%-1%
            trainingDatasetSize = static_cast<int>(datasetSize * 0.98);
            validationDatasetSize = static_cast<int>(datasetSize * 0.01);
        }

        DatasetFormat trainingDataset(datasetCopy.begin() + 0, datasetCopy.begin() + trainingDatasetSize);
        DatasetFormat validationDataset(datasetCopy.begin() + trainingDatasetSize,
                                        datasetCopy.begin() + trainingDatasetSize + validationDatasetSize);
        DatasetFormat testDataset(datasetCopy.begin() + trainingDatasetSize + validationDatasetSize, datasetCopy.end());

        auto data = std::make_tuple(trainingDataset, validationDataset, testDataset);

        return data;
    }

    /**
     * @brief Train the network on the provided dataset
     *
     * @param dataset - vector of (input_vector, target_label) pairs
     * @return average loss after training
     */
    double train(const DatasetFormat &dataset) {
        if (dataset.empty()) {
            throw std::invalid_argument("Dataset cannot be empty");
        }

        if (!network) {
            throw std::invalid_argument("Network pointer is null");
        }

        const auto datasets = splitData(dataset);
        const auto trainingDataset = std::get<0>(datasets);
        const auto validationDataset = std::get<1>(datasets);
        const auto testDataset = std::get<2>(datasets);

        if (verbose) {
            std::cout << "Training for " << epochs << " epochs..." << std::endl;
            std::cout << "Total Data: " << dataset.size() << std::endl;
            std::cout << "Training Data: " << trainingDataset.size()
                    << " | Validation Data: " << validationDataset.size()
                    << " | Test Data: " << testDataset.size() << std::endl;
            std::cout << "Batch size: " << batchSize << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
        }

        double finalTrainingLoss = 0.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epochLoss = 0.0;
            int sampleCount = 0;

            // Accumulator for gradient averaging within a batch
            std::vector<double> accumulatedGradients(network->parameters().size(), 0.0);

            for (const auto &sample: trainingDataset) {
                const auto &inputs = sample.first;
                const double target = sample.second;

                auto inputNodes = helper::createInputNodes(inputs);

                // Forward pass
                std::vector<Node *> predictions = (*network)(inputNodes);

                // Compute loss
                Node *loss = lossFunction(predictions, target);
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

                // Update parameters when batch is full or at end of the training dataset
                if (sampleCount % batchSize == 0 || sampleCount == static_cast<int>(trainingDataset.size())) {
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

            finalTrainingLoss = epochLoss / static_cast<double>(trainingDataset.size());

            // Print progress
            if (verbose && (epoch + 1) % printEvery == 0) {
                // Calculate accuracy on the Validation set
                const double accuracy = computeAccuracy(validationDataset);

                std::cout << "Epoch " << std::setw(4) << (epoch + 1)
                        << " | Loss: " << std::fixed << std::setprecision(6) << finalTrainingLoss
                        << " | Accuracy: " << std::setprecision(2) << (accuracy * 100.0) << "%"
                        << std::endl;
            }
        }

        if (verbose) {
            std::cout << "\nTraining complete!" << std::endl;
            std::cout << "Evaluating on Test Set..." << std::endl;

            const double testAccuracy = computeAccuracy(testDataset);
            std::cout << "Final Test Set Accuracy: " << std::fixed << std::setprecision(2)
                    << (testAccuracy * 100.0) << "%" << std::endl;
        }

        return finalTrainingLoss;
    }
};

#endif //TRAINER_H

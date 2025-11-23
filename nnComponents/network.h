#ifndef NETWORK_H
#define NETWORK_H
#include <nnComponents/module.h>
#include <nnComponents/layer.h>
#include <nnComponents/neuron.h>
#include <utils/serialization/modelSerializer.h>

/**
 * The network class provides the user with the right amount of flexibility to customize their own Multi-Layer Perceptron,
 * train it with the desired training parameters, save it a in a binary file and finally use it to make inferences.
 */
class Network : public Module {
protected:
    std::vector<std::pair<int, Activation> > networkSpecs;
    std::vector<Layer> layers;

public:
    Network() = default;

    /**
     * @brief Creates the default network with ReLU activation function
     *
     * @param networkSpecs - holds information about the size and type of each layer
     */
    explicit Network(const std::vector<std::pair<int, Activation> > &networkSpecs) : networkSpecs(networkSpecs) {
        for (size_t i = 0; i < networkSpecs.size() - 1; i++) {
            layers.emplace_back(networkSpecs.at(i).first, networkSpecs.at(i + 1).first, networkSpecs.at(i + 1).second);
        }
    }

    /**
     * @brief Performs the forward-passing functionality of the network
     *
     * @param inputVector
     * @return A vector of Node object pointers that point to the activations of the output layer
     */
    virtual std::vector<Node *> operator()(const std::vector<Node *> &inputVector) {
        std::vector<Node *> x = inputVector;
        for (auto &layer: layers) {
            // The output vector of a layer becomes the input vector of the other
            x = layer(x);
        }
        return x;
    }

    /**
     * @brief Flattens all the parameters of the network into a single one-dimensional vector.
     *
     * @return A vector of Node object pointers that point to the parameters of the network
     */
    std::vector<Node *> parameters() override {
        // Flattens all the parameters of a layer so that they are accessible through this vector
        std::vector<Node *> params;
        for (auto &layer: layers) {
            auto layerParameters = layer.parameters();
            params.insert(params.end(), layerParameters.begin(), layerParameters.end());
        }
        return params;
    }

    /**
     * @brief Returns a string that contains information about the types of the layers that
     * comprise the network along with the size of each layer.
     *
     * @return A string that gives a general overview of the network structure.
     */
    virtual std::string representation() const {
        std::string s = "Network of [";
        for (size_t i = 0; i < layers.size(); ++i) {
            s += layers.at(i).representation();
            if (i + 1 < layers.size()) s += ", ";
        }
        return s + "]";
    }

    /**
     * @brief Provides training logic
     *
     * @param learningRate - The size of the optimizer step, which directly affects learning
     * @param epochs - The amount of times the training driver goes through all the training data
     * @param batchSize - The amount of samples processed before we have an optimizer step
     * @param dataset - The data that the network is going to use to train
     */
    virtual void train(double learningRate, int epochs, int batchSize,
                       std::vector<std::pair<std::vector<double>, double> > &dataset) = 0;


    /**
     * Method used to make inferences
     *
     * @param input - The input vector tha goes into the model to make inferences
     * @return Returns the number of the predicted category
     */
    virtual int predict(std::vector<double> &input) = 0;

    /**
     * @brief Provides sufficient information for the library to load a pre-trained model into memory
     *
     * @return ModelMetadata - The modelling of the data that is needed to load a trained model into memory and use it
     * to make inferences.
     */
    virtual ModelMetadata getMetadata() {
        std::vector<int> hiddenLayerSizes;
        for (size_t i = 1; i < networkSpecs.size() - 1; i++) {
            hiddenLayerSizes.emplace_back(networkSpecs.at(i).first);
        }
        return ModelMetadata{this->networkSpecs.at(0).first, hiddenLayerSizes, parameters().size()};
    }

    /**
     * @brief Saves the model parameters and metadata in persistent memory so that it can be loaded and used to make
     * inferences.
     *
     * @param filepath - Specifies that filepath where the binary file with the model specs will be saved
     * @return returns True if the model was saved successfully and false otherwise.
     */
    virtual bool saveModel(const std::string &filepath) {
        const std::vector<Node *> params = this->parameters();
        const ModelMetadata metadata = getMetadata();

        return ModelSerializer::saveWithMetadata(params, metadata, filepath);
    }
};

/// The overloading function of the operator << that calls the representation() method of the network for convenience
inline std::ostream &operator<<(std::ostream &os, const Network &m) {
    return os << m.representation();
}

#endif //NETWORK_H

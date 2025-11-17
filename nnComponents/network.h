#ifndef NETWORK_H
#define NETWORK_H
#include <nnComponents/module.h>
#include <nnComponents/layer.h>
#include <nnComponents/neuron.h>

class Network : public Module {
protected:
    std::vector<Layer> layers;
public:
    Network() = default;

    /**
     * @brief Creates the default network with ReLU activation function
     *
     * @param numberOfInputs - Marks the dimension of the input flattened into a one-dimensional vector
     * @param hiddenLayerSizes - A vector containing the size of the hidden layers of the network
     */
    Network(const int numberOfInputs, const std::vector<int> &hiddenLayerSizes) {
        // The vector below holds the dimensions of each layer of the network
        std::vector<int> networkDimensions;
        networkDimensions.reserve(hiddenLayerSizes.size() + 1); // to account for the input layer size
        networkDimensions.push_back(numberOfInputs);
        networkDimensions.insert(networkDimensions.end(), hiddenLayerSizes.begin(), hiddenLayerSizes.end());

        for (size_t i = 0; i < hiddenLayerSizes.size(); ++i) {
            layers.emplace_back(networkDimensions.at(i), networkDimensions.at(i + 1), Activation::RELU);
        }
    }

    /**
     * @brief Performs the forward-passing functionality of the network
     *
     * @param inputVector
     * @return A vector of Node object pointers that point to the activations of the output layer
     */
    std::vector<Node *> operator()(const std::vector<Node *> &inputVector) {
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
     * @brief The function returns a string that contains information about the types of the layers that
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

    virtual void train(const double learningRate, const int epochs, const int batchSize, std::vector<std::pair<std::vector<double>, double>> dataset) {}
};

/// The overloading function of the operator << that calls the representation() method of the network for convenience
inline std::ostream &operator<<(std::ostream &os, const Network &m) {
    return os << m.representation();
}

#endif //NETWORK_H

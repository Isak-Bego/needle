#ifndef MODELSERIALIZER_H
#define MODELSERIALIZER_H

#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <autoGradEngine/node.h>

struct ModelMetadata {
    int inputVectorSize;
    std::vector<int> hiddenLayerSizes;
    size_t totalParameters;

    ModelMetadata() : inputVectorSize(0), totalParameters(0) {}

    ModelMetadata(const int inputs, const std::vector<int>& hiddenSizes, const size_t numParams)
        : inputVectorSize(inputs), hiddenLayerSizes(hiddenSizes), totalParameters(numParams) {}
};

class ModelSerializer {
public:
    // Save model with metadata
    static bool saveWithMetadata(const std::vector<Node*>& parameters,
                                   const ModelMetadata& metadata,
                                   const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Write metadata
        file.write(reinterpret_cast<const char*>(&metadata.inputVectorSize), sizeof(metadata.inputVectorSize));

        size_t hidden_size = metadata.hiddenLayerSizes.size();
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));

        for (int layer_size : metadata.hiddenLayerSizes) {
            file.write(reinterpret_cast<const char*>(&layer_size), sizeof(layer_size));
        }

        file.write(reinterpret_cast<const char*>(&metadata.totalParameters), sizeof(metadata.totalParameters));

        // Write parameters
        size_t num_params = parameters.size();
        file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

        for (const Node* param : parameters) {
            if (param) {
                file.write(reinterpret_cast<const char*>(&param->data), sizeof(param->data));
            }
        }

        file.close();
        return true;
    }

    // Load model metadata without loading parameters
    static ModelMetadata loadMetadata(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        ModelMetadata metadata;

        // Read metadata
        file.read(reinterpret_cast<char*>(&metadata.inputVectorSize), sizeof(metadata.inputVectorSize));

        size_t hidden_size = 0;
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));

        metadata.hiddenLayerSizes.resize(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            file.read(reinterpret_cast<char*>(&metadata.hiddenLayerSizes.at(i)),
                     sizeof(metadata.hiddenLayerSizes.at(i)));
        }

        file.read(reinterpret_cast<char*>(&metadata.totalParameters), sizeof(metadata.totalParameters));

        file.close();
        return metadata;
    }

    // Load parameters with validation
    static bool loadWithValidation(std::vector<Node*>& parameters,
                                    const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Skip metadata section
        ModelMetadata metadata;
        file.read(reinterpret_cast<char*>(&metadata.inputVectorSize), sizeof(metadata.inputVectorSize));

        size_t hidden_size = 0;
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));

        for (size_t i = 0; i < hidden_size; ++i) {
            int dummy;
            file.read(reinterpret_cast<char*>(&dummy), sizeof(dummy));
        }

        file.read(reinterpret_cast<char*>(&metadata.totalParameters), sizeof(metadata.totalParameters));

        // Read parameters
        size_t num_params = 0;
        file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

        if (num_params != parameters.size()) {
            file.close();
            throw std::runtime_error(
                "Parameter count mismatch: file has " + std::to_string(num_params) +
                " parameters but model has " + std::to_string(parameters.size())
            );
        }

        for (Node* param : parameters) {
            if (param) {
                file.read(reinterpret_cast<char*>(&param->data), sizeof(param->data));
            }
        }

        file.close();
        return true;
    }
};

#endif //MODELSERIALIZER_H
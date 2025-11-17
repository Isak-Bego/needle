#ifndef MODELSERIALIZER_H
#define MODELSERIALIZER_H

#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <autoGradEngine/node.h>

struct ModelMetadata {
    int num_inputs;
    std::vector<int> hidden_layer_sizes;
    size_t total_parameters;

    ModelMetadata() : num_inputs(0), total_parameters(0) {}

    ModelMetadata(int inputs, const std::vector<int>& hidden_sizes, size_t num_params)
        : num_inputs(inputs), hidden_layer_sizes(hidden_sizes), total_parameters(num_params) {}
};

class ModelSerializer {
public:
    // Save model with metadata
    static bool save_with_metadata(const std::vector<Node*>& parameters,
                                   const ModelMetadata& metadata,
                                   const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Write metadata
        file.write(reinterpret_cast<const char*>(&metadata.num_inputs), sizeof(metadata.num_inputs));

        size_t hidden_size = metadata.hidden_layer_sizes.size();
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));

        for (int layer_size : metadata.hidden_layer_sizes) {
            file.write(reinterpret_cast<const char*>(&layer_size), sizeof(layer_size));
        }

        file.write(reinterpret_cast<const char*>(&metadata.total_parameters), sizeof(metadata.total_parameters));

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
    static ModelMetadata load_metadata(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        ModelMetadata metadata;

        // Read metadata
        file.read(reinterpret_cast<char*>(&metadata.num_inputs), sizeof(metadata.num_inputs));

        size_t hidden_size = 0;
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));

        metadata.hidden_layer_sizes.resize(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            file.read(reinterpret_cast<char*>(&metadata.hidden_layer_sizes.at(i)),
                     sizeof(metadata.hidden_layer_sizes.at(i)));
        }

        file.read(reinterpret_cast<char*>(&metadata.total_parameters), sizeof(metadata.total_parameters));

        file.close();
        return metadata;
    }

    // Load parameters with validation
    static bool load_with_validation(std::vector<Node*>& parameters,
                                    const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Skip metadata section
        ModelMetadata metadata;
        file.read(reinterpret_cast<char*>(&metadata.num_inputs), sizeof(metadata.num_inputs));

        size_t hidden_size = 0;
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));

        for (size_t i = 0; i < hidden_size; ++i) {
            int dummy;
            file.read(reinterpret_cast<char*>(&dummy), sizeof(dummy));
        }

        file.read(reinterpret_cast<char*>(&metadata.total_parameters), sizeof(metadata.total_parameters));

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
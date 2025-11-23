#ifndef MODELSERIALIZER_H
#define MODELSERIALIZER_H

#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <limits>
#include <autoGradEngine/node.h>

struct ModelMetadata {
    int inputVectorSize;
    std::vector<int> hiddenLayerSizes;
    size_t totalParameters;

    ModelMetadata() : inputVectorSize(0), totalParameters(0) {
    }

    ModelMetadata(const int inputs, const std::vector<int> &hiddenSizes, const size_t numParams)
        : inputVectorSize(inputs), hiddenLayerSizes(hiddenSizes), totalParameters(numParams) {
    }
};

/**
 * The model serializer captures the parameters of a network along with its layer structure and saves them in a .txt file.
 * Additionally, it offers methods for retrieving that information for creating inference models that help making predictions.
 */
class ModelSerializer {
public:
    // Save model with metadata in text format
    static bool saveWithMetadata(const std::vector<Node *> &parameters,
                                 const ModelMetadata &metadata,
                                 const std::string &filepath) {
        std::ofstream file(filepath); // Text mode by default
        if (!file.is_open()) {
            return false;
        }

        // Write metadata
        file << metadata.inputVectorSize << "\n";

        size_t hidden_size = metadata.hiddenLayerSizes.size();
        file << hidden_size << "\n";

        for (int layer_size: metadata.hiddenLayerSizes) {
            file << layer_size << " ";
        }
        file << "\n";

        file << metadata.totalParameters << "\n";

        // Write parameters
        size_t num_params = parameters.size();
        file << num_params << "\n";

        // Use maximum precision to avoid losing accuracy in text conversion
        file << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);

        for (const Node *param: parameters) {
            if (param) {
                file << param->data << "\n";
            }
        }

        file.close();
        return true;
    }

    // Load model metadata without loading parameters
    static ModelMetadata loadMetadata(const std::string &filepath) {
        std::ifstream file(filepath); // Text mode by default
        if (!file.is_open()) {
            std::cout << "Failed to open file: " + filepath << std::endl;
            return ModelMetadata();
        }

        ModelMetadata metadata;

        // Read metadata
        if (!(file >> metadata.inputVectorSize)) return ModelMetadata();

        size_t hidden_size = 0;
        file >> hidden_size;

        metadata.hiddenLayerSizes.resize(hidden_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            file >> metadata.hiddenLayerSizes.at(i);
        }

        file >> metadata.totalParameters;

        file.close();
        return metadata;
    }

    // Load parameters with validation
    static bool loadWithValidation(std::vector<Node *> &parameters,
                                   const std::string &filepath) {
        std::ifstream file(filepath); // Text mode by default
        if (!file.is_open()) {
            return false;
        }

        // Skip metadata section
        int inputVectorSize;
        file >> inputVectorSize;

        size_t hidden_size = 0;
        file >> hidden_size;

        for (size_t i = 0; i < hidden_size; ++i) {
            int dummy;
            file >> dummy;
        }

        size_t totalParameters;
        file >> totalParameters;

        // Read parameters count
        size_t num_params = 0;
        file >> num_params;

        if (num_params != parameters.size()) {
            file.close();
            std::cout <<
                "Parameter count mismatch: file has " + std::to_string(num_params) +
                " parameters but model has " + std::to_string(parameters.size()) << std::endl;
            return false;
        }

        // Read parameter values
        for (Node *param: parameters) {
            if (param) {
                file >> param->data;
            }
        }

        file.close();
        return true;
    }
};

#endif //MODELSERIALIZER_H
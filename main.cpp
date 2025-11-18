#include <vector>
#include <iostream>
#include <models/multiClassClassifier.h>
#include <utils/datasets/irisDataset.h>

int main() {

    std::cout << "=== Training New Multi-Class Model ===" << std::endl;
    auto dataset = IrisDataset::get_data();

    // Create a multi-class classifier
    // 4 inputs (features), hidden layers of [8, 8], 3 output classes
    MultiClassClassifier model(
        IrisDataset::get_num_features(),
        {8, 8},
        IrisDataset::get_num_classes()
    );

    std::cout << model << std::endl;
    std::cout << std::endl;

    model.train(0.1, 2000, 100, dataset);

    // Save the trained model with metadata
    if (model.saveModel("iris_model.bin")) {
        std::cout << "\n✓ Model saved successfully to 'iris_model.bin'" << std::endl;
    } else {
        std::cerr << "\n✗ Failed to save model" << std::endl;
    }

    std::cout << "\n=================================\n" << std::endl;


    std::cout << "=== Loading Model from File ===" << std::endl;

    // The architecture is automatically reconstructed from the file
    MultiClassClassifier* loaded_model = MultiClassClassifier::loadFromFile("iris_model.bin");

    if (loaded_model != nullptr) {
        std::cout << "\n=== Testing Loaded Model ===" << std::endl;

        // Test the loaded model with some samples
        std::vector<std::vector<double>> test_inputs = {
            {0.22, 0.63, 0.07, 0.04},  // Should predict Setosa (0)
            {0.69, 0.42, 0.51, 0.38},  // Should predict Versicolor (1)
            {0.72, 0.50, 0.69, 0.67},  // Should predict Virginica (2)
            {0.17, 0.42, 0.07, 0.04},  // Should predict Setosa (0)
            {0.50, 0.25, 0.46, 0.38},  // Should predict Versicolor (1)
        };

        auto class_names = IrisDataset::get_class_names();

        for (auto& inputs : test_inputs) {
            int prediction = loaded_model->predict(inputs);
            auto probabilities = loaded_model->predict_proba(inputs);

            std::cout << "Input: [";
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::cout << inputs[i];
                if (i + 1 < inputs.size()) std::cout << ", ";
            }
            std::cout << "]" << std::endl;

            std::cout << "  -> Prediction: " << prediction
                     << " (" << class_names[prediction] << ")" << std::endl;

            std::cout << "  -> Probabilities: [";
            for (size_t i = 0; i < probabilities.size(); ++i) {
                std::cout << std::fixed << std::setprecision(4) << probabilities[i];
                if (i + 1 < probabilities.size()) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << std::endl;
        }

        // Clean up
        delete loaded_model;
    } else {
        std::cerr << "✗ Failed to load model" << std::endl;
    }

    std::cout << "=================================\n" << std::endl;

    return 0;
}
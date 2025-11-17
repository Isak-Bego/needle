#include <vector>
#include <models/binaryClassifier.h>
#include <utils/datasets/xorDataset.h>

int main() {

    std::cout << "=== Training New Model ===" << std::endl;
    auto dataset = XORDataset::get_data();
    BinaryClassifier model(2, {8, 8});

    model.train(0.2, 1000, 1, dataset);

    // Save the trained model with metadata
    if (model.save_model("xor_model.bin")) {
        std::cout << "\n✓ Model saved successfully to 'xor_model.bin'" << std::endl;
    } else {
        std::cerr << "\n✗ Failed to save model" << std::endl;
    }

    std::cout << "\n=================================\n" << std::endl;


    std::cout << "=== Loading Model from File ===" << std::endl;

    // The architecture is automatically reconstructed from the file
    BinaryClassifier* loaded_model = BinaryClassifier::load_from_file("xor_model.bin");

    if (loaded_model != nullptr) {
        std::cout << "\n=== Testing Loaded Model ===" << std::endl;

        // Test the loaded model
        std::vector<std::vector<double>> test_inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0},
            {0.0, 1.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {0.0, 0.0}
        };

        for (auto& inputs : test_inputs) {
            int prediction = loaded_model->predict(inputs);
            std::cout << "Input: [" << inputs[0] << ", " << inputs[1] << "] "
                     << "-> Prediction: " << prediction << std::endl;
        }

        // Clean up
        delete loaded_model;
    } else {
        std::cerr << "✗ Failed to load model" << std::endl;
    }

    std::cout << "\n=================================\n" << std::endl;

    return 0;
}
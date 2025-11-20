#ifndef IRISDATASET_H
#define IRISDATASET_H

#include <vector>
#include <algorithm>  // std::shuffle
#include <random>     // std::mt19937, std::random_device

class IrisDataset {
public:
    // Simplified Iris dataset with 4 features and 3 classes
    // Features: [sepal_length, sepal_width, petal_length, petal_width]
    // Classes: 0 = Setosa, 1 = Versicolor, 2 = Virginica

    static std::vector<std::pair<std::vector<double>, double> > get_data() {
        std::vector<std::pair<std::vector<double>, double> > data;

        // Class 0: Setosa (normalized values)
        data.push_back({{0.22, 0.63, 0.07, 0.04}, 0.0});
        data.push_back({{0.17, 0.42, 0.07, 0.04}, 0.0});
        data.push_back({{0.11, 0.50, 0.05, 0.04}, 0.0});
        data.push_back({{0.08, 0.46, 0.08, 0.04}, 0.0});
        data.push_back({{0.19, 0.67, 0.07, 0.04}, 0.0});
        data.push_back({{0.31, 0.79, 0.10, 0.08}, 0.0});
        data.push_back({{0.08, 0.58, 0.07, 0.08}, 0.0});
        data.push_back({{0.19, 0.58, 0.08, 0.04}, 0.0});
        data.push_back({{0.03, 0.38, 0.07, 0.04}, 0.0});
        data.push_back({{0.17, 0.46, 0.08, 0.00}, 0.0});

        // Class 1: Versicolor
        data.push_back({{0.69, 0.42, 0.51, 0.38}, 1.0});
        data.push_back({{0.50, 0.25, 0.46, 0.38}, 1.0});
        data.push_back({{0.69, 0.50, 0.59, 0.50}, 1.0});
        data.push_back({{0.42, 0.29, 0.49, 0.46}, 1.0});
        data.push_back({{0.58, 0.50, 0.49, 0.42}, 1.0});
        data.push_back({{0.53, 0.38, 0.51, 0.42}, 1.0});
        data.push_back({{0.47, 0.42, 0.46, 0.42}, 1.0});
        data.push_back({{0.67, 0.46, 0.56, 0.42}, 1.0});
        data.push_back({{0.56, 0.46, 0.49, 0.38}, 1.0});
        data.push_back({{0.50, 0.33, 0.46, 0.38}, 1.0});

        // Class 2: Virginica
        data.push_back({{0.72, 0.50, 0.69, 0.67}, 2.0});
        data.push_back({{0.58, 0.42, 0.63, 0.75}, 2.0});
        data.push_back({{0.75, 0.50, 0.76, 0.75}, 2.0});
        data.push_back({{0.64, 0.42, 0.68, 0.67}, 2.0});
        data.push_back({{0.78, 0.58, 0.81, 0.83}, 2.0});
        data.push_back({{0.86, 0.42, 0.85, 0.92}, 2.0});
        data.push_back({{0.64, 0.33, 0.69, 0.58}, 2.0});
        data.push_back({{0.72, 0.42, 0.75, 0.83}, 2.0});
        data.push_back({{0.69, 0.38, 0.71, 0.75}, 2.0});
        data.push_back({{0.67, 0.46, 0.68, 0.71}, 2.0});

        // Shuffle the dataset
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data.begin(), data.end(), g);

        return data;
    }


    // Returns the full dataset repeated n times (for multiple epochs)
    static std::vector<std::pair<std::vector<double>, double> > get_repeated(int n) {
        auto base_data = get_data();
        std::vector<std::pair<std::vector<double>, double> > repeated;
        repeated.reserve(base_data.size() * n);

        for (int i = 0; i < n; ++i) {
            repeated.insert(repeated.end(), base_data.begin(), base_data.end());
        }

        return repeated;
    }

    // Get number of features
    static int get_num_features() {
        return 4;
    }

    // Get number of classes
    static int get_num_classes() {
        return 3;
    }

    // Get class names for interpretation
    static std::vector<std::string> get_class_names() {
        return {"Setosa", "Versicolor", "Virginica"};
    }
};

#endif //IRISDATASET_H

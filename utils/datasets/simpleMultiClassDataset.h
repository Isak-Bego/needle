#ifndef SIMPLEMULTICLASSDATASET_H
#define SIMPLEMULTICLASSDATASET_H

#include <vector>
#include <utility>

class SimpleMultiClassDataset {
public:
    // Simple 3-class classification problem with 2D inputs
    // Class 0: points around (0, 0)
    // Class 1: points around (1, 1)
    // Class 2: points around (1, 0)

    static std::vector<std::pair<std::vector<double>, double> > get_data() {
        std::vector<std::pair<std::vector<double>, double> > data;

        // Class 0: Lower left region
        data.push_back({{0.0, 0.0}, 0.0});
        data.push_back({{0.1, 0.1}, 0.0});
        data.push_back({{0.0, 0.2}, 0.0});
        data.push_back({{0.2, 0.0}, 0.0});
        data.push_back({{0.1, 0.0}, 0.0});

        // Class 1: Upper right region
        data.push_back({{1.0, 1.0}, 1.0});
        data.push_back({{0.9, 0.9}, 1.0});
        data.push_back({{1.0, 0.8}, 1.0});
        data.push_back({{0.8, 1.0}, 1.0});
        data.push_back({{0.9, 1.0}, 1.0});

        // Class 2: Lower right region
        data.push_back({{1.0, 0.0}, 2.0});
        data.push_back({{0.9, 0.1}, 2.0});
        data.push_back({{1.0, 0.2}, 2.0});
        data.push_back({{0.8, 0.0}, 2.0});
        data.push_back({{0.9, 0.0}, 2.0});

        return data;
    }

    static std::vector<std::pair<std::vector<double>, double> > get_repeated(int n) {
        auto base_data = get_data();
        std::vector<std::pair<std::vector<double>, double> > repeated;
        repeated.reserve(base_data.size() * n);

        for (int i = 0; i < n; ++i) {
            repeated.insert(repeated.end(), base_data.begin(), base_data.end());
        }

        return repeated;
    }

    static int get_num_features() {
        return 2;
    }

    static int get_num_classes() {
        return 3;
    }

    static std::vector<std::string> get_class_names() {
        return {"Class A", "Class B", "Class C"};
    }
};

#endif //SIMPLEMULTICLASSDATASET_H

#ifndef XORDATASET_H
#define XORDATASET_H
#include <vector>
#include <utility>

class XORDataset {
public:
    // Returns pairs of (input_vector, label)
    static std::vector<std::pair<std::vector<double>, double>> get_data() {
        std::vector<std::pair<std::vector<double>, double>> data;

        // XOR truth table
        // Input: [x1, x2] -> Output: x1 XOR x2
        data.push_back({{0.0, 0.0}, 0.0});  // 0 XOR 0 = 0
        data.push_back({{0.0, 1.0}, 1.0});  // 0 XOR 1 = 1
        data.push_back({{1.0, 0.0}, 1.0});  // 1 XOR 0 = 1
        data.push_back({{1.0, 1.0}, 0.0});  // 1 XOR 1 = 0

        return data;
    }

    // Returns the full dataset repeated n times (for multiple epochs)
    static std::vector<std::pair<std::vector<double>, double>> get_repeated(int n) {
        auto base_data = get_data();
        std::vector<std::pair<std::vector<double>, double>> repeated;
        repeated.reserve(base_data.size() * n);

        for (int i = 0; i < n; ++i) {
            repeated.insert(repeated.end(), base_data.begin(), base_data.end());
        }

        return repeated;
    }
};

#endif //XORDATASET_H

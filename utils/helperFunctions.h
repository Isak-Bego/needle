#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H
#include <vector>
#include <autoGradEngine/node.h>

namespace helper {
    template<typename T>
    int find(std::vector<T> &vec, T val) {
        for (std::size_t i = 0; i < vec.size(); i++) {
            if (vec[i] == val) {
                return i;
            }
        }
        return -1;
    }

    inline std::vector<Node*> createInputNodes(const std::vector<double>& inputs) {
        std::vector<Node*> input_nodes;
        input_nodes.reserve(inputs.size());
        for (const double val : inputs) {
            input_nodes.push_back(new Node(val));
        }
        return input_nodes;
    }

    inline void deleteInputNodes(std::vector<Node*> &nodes) {
        for (const Node* node : nodes) {
            delete node;
        }
    }
}

#endif //HELPERFUNCTIONS_H

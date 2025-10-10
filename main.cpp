#include <iostream>
#include "layers/layer.h"
#include "layers/network.h"


class Node {
    float value = 0.0f;
    Node *left = nullptr;
    Node *right = nullptr;
    char operation = '\0';
    bool var = false;
    Neuron* neuron_link = nullptr;

public:
    explicit Node (float value) {
        this->value = value;
    }

    Node(float value, Node *left, Node *right, char operation) {
        this->value = value;
        this->left = left;
        this->right = right;
        this->operation = operation;
    }

    // Copy constructor
    Node(const Node &n) {
        this->value = n.value;
        this->left = n.left;
        this->right = n.right;
        this->operation = n.operation;
    }

    Node operator+(Node &right) {
        Node* temp = new Node(*this);
        return Node(this->value + right.value, temp, &right, '+');
    }

    Node operator*(Node &right) {
        Node* temp = new Node(*this);
        return Node(this->value * right.value, temp, &right, '*');
    }

    float get_value() {
        return value;
    }

    Node& get_left() {
        return *left;
    }

    Node& get_right() {
        return *right;
    }

    char get_operation() {
        return operation;
    }
};



int main() {
    std::vector<float> inputs = {2.4, 3.5, 4.7, 2.3, 5.8, 9.4, 2.3, 4.6};
    std::vector<float> weights = {3, 3, 3, 3, 3 ,3 ,3, 3};
    std::vector<Node> inputNodes;
    std::vector<Node> weightNodes;


    for (auto in : inputs) {
        inputNodes.emplace_back(in);
    }

    for (auto w : weights) {
        weightNodes.emplace_back(w);
    }

    Node nodeSum = Node(0.0f);
    float sum = 0.0f;
    for(auto i = 0; i < inputNodes.size(); i++) {
        Node prod = inputNodes.at(i) * weightNodes.at(i);
        sum += inputs.at(i) * weights.at(i);
        nodeSum = nodeSum + prod;
    }

    std::cout<<nodeSum.get_value()<<std::endl;
    std::cout<<sum<<std::endl;
    // std::vector<std::pair<std::vector<float>, float>> trainingData = {{{1, 2, 3}, 7}, {{4, 5, 6}, 8}, {{7, 8, 9}, 9}, {{2, 4, 6}, 12}};
    //
    // Network net = Network({3, 2, 3});
    // net.loadTrainingData(trainingData);
    // net.forwardPass();
    // net.printLayers();
    //
    // std::cout<<std::endl<<"The mean squared error is: "<<net.getMeanSquaredError();
    return 0;
}

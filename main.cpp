#include <iostream>
#include "layers/layer.h"
#include "layers/network.h"
#include <stack>
#include <tuple>

// Auto-forward pass
class Node {
    float value = 0.0f;
    Node *left = nullptr;
    Node *right = nullptr;
    char operation = '\0';
    bool isVisited = false;
    float gradient = 0.0f;
    bool var = false;
    Neuron *neuron_link = nullptr;

public:
    explicit Node(float value, bool isVariable) {
        this->value = value;
        this->var = isVariable;
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
        this->isVisited = n.isVisited;
        this->var = n.var;
        this->gradient = n.gradient;
    }

    Node operator+(Node &right) {
        Node *temp = new Node(*this);
        return Node(this->value + right.value, temp, &right, '+');
    }

    Node operator*(Node &right) {
        Node *temp = new Node(*this);
        return Node(this->value * right.value, temp, &right, '*');
    }

    float get_value() {
        return value;
    }

    Node *get_left() {
        return left;
    }

    Node *get_right() {
        return right;
    }

    float get_gradient() {
        return gradient;
    }

    char get_operation() {
        return operation;
    }

    void computePartials() {
        // 0) Reset flags/gradients so repeated calls work predictably
        {
            std::stack<Node *> st;
            st.push(this);
            while (!st.empty()) {
                Node *n = st.top();
                st.pop();
                n->isVisited = false;
                if (n->var) n->gradient = 0.0f;
                if (n->right) st.push(n->right);
                if (n->left) st.push(n->left);
            }
        }

        // 1) Reverse-mode pass using an explicit stack
        std::stack<std::tuple<Node*, float> > nodeStack;
        nodeStack.emplace(this, 1.0f);

        while (!nodeStack.empty()) {
            auto tempNode = std::get<0>(nodeStack.top());
            auto tempSeed = std::get<1>(nodeStack.top());
            nodeStack.pop();

            // Accumulate gradient if this node is a leaf variable
            if (tempNode->var) {
                std::cout<<"Seed: "<<tempSeed<<" ";
                tempNode->gradient += tempSeed;
                std::cout<<"Gradiant: "<<tempNode->gradient<<std::endl;
            }

            // Push children exactly once per node
            if (!tempNode->isVisited) {
                Node *L = tempNode->left;
                Node *R = tempNode->right;

                switch (tempNode->operation) {
                    case '+':
                        if (R) nodeStack.emplace(R, tempSeed);
                        if (L) nodeStack.emplace(L, tempSeed);
                        break;
                    case '*':
                        if (R) nodeStack.emplace(R, (L ? L->value : 0.0f) * tempSeed);
                        if (L) nodeStack.emplace(L, (R ? R->value : 0.0f) * tempSeed);
                        break;
                    default:
                        break;
                }

                tempNode->isVisited = true;
            }
        }
    }
};


int main() {
    Node a = Node(3.0, true);
    Node b = Node(3.0, true);
    Node c = Node(5.0, true);
    Node d = Node(4.0, true);

    Node e = a + b;
    e = e * c;
    e = e + d;

    e.computePartials();

    std::cout << a.get_gradient()<< std::endl;
    std::cout << b.get_gradient()<< std::endl;
    std::cout << c.get_gradient()<< std::endl;
    std::cout << d.get_gradient()<< std::endl;

    // std::vector<float> inputs = {2.4, 3.5, 4.7, 2.3, 5.8, 9.4, 2.3, 4.6};
    // std::vector<float> weights = {3, 3, 3, 3, 3 ,3 ,3, 3};
    // std::vector<Node> inputNodes;
    // std::vector<Node> weightNodes;
    //
    //
    // for (auto in : inputs) {
    //     inputNodes.emplace_back(in);
    // }
    //
    // for (auto w : weights) {
    //     weightNodes.emplace_back(w);
    // }
    //
    // Node nodeSum = Node(0.0f);
    // float sum = 0.0f;
    // for(auto i = 0; i < inputNodes.size(); i++) {
    //     Node prod = inputNodes.at(i) * weightNodes.at(i);
    //     sum += inputs.at(i) * weights.at(i);
    //     nodeSum = nodeSum + prod;
    // }
    //
    // std::cout<<nodeSum.get_value()<<std::endl;
    // std::cout<<sum<<std::endl;
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

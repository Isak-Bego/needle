#ifndef EXPRESSIONNODE_H
#define EXPRESSIONNODE_H

#include "utils/activation-functions/sigmoid.h"
#include "utils/activation-functions/softmax.h"
#include <stack>
#include <tuple>
#include <vector>

// Auto-forward pass
class Node {
    double value = 0.0;
    Node *left = nullptr;
    Node *right = nullptr;
    char operation = '\0';
    bool isVisited = false;
    double gradient = 0.0;
    bool var = false;
    bool isSigmoidApplied = false;
    bool isSoftmaxApplied = false;
    std::vector<Node *> ownedNodes; // Track nodes we created

    // Softmax-specific data
    std::vector<Node*> softmaxGroup;  // All nodes in the same softmax operation
    std::vector<double> softmaxOutputs;  // Cached softmax outputs
    int softmaxIndex = -1;  // This node's index in the softmax group

public:
    explicit Node(const double value, const bool isVariable) {
        this->value = value;
        this->var = isVariable;
    }

    ~Node() {
        for (const Node *n: ownedNodes) {
            delete n;
        }
    }

    Node(const double value, Node *left, Node *right, const char operation) {
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
        this->isSigmoidApplied = n.isSigmoidApplied;
        this->isSoftmaxApplied = n.isSoftmaxApplied;
        this->softmaxGroup = n.softmaxGroup;
        this->softmaxOutputs = n.softmaxOutputs;
        this->softmaxIndex = n.softmaxIndex;
    }

    Node* operator+(Node &right) {
        auto *result = new Node(this->value + right.value, this, &right, '+');
        ownedNodes.push_back(result);
        return result;
    }

    Node* operator*(Node &right) {
        auto *result = new Node(this->value * right.value, this, &right, '*');
        ownedNodes.push_back(result);
        return result;
    }

    Node* operator/(Node& right) {
        auto* result = new Node(this->value / right.value, this, &right, '/');
        ownedNodes.push_back(result);
        return result;
    }

    double get_value() const {
        return value;
    }

    void set_value(const double &tempVal) {
        this->value = tempVal;
    }

    Node *get_left() const{
        return left;
    }

    Node *get_right() const{
        return right;
    }

    double get_gradient() const{
        return gradient;
    }

    char get_operation() const{
        return operation;
    }

    void apply_sigmoid() {
        this->value = sigmoid(this->value);
        this->isSigmoidApplied = true;
    }

    /**
     * Apply softmax to a group of nodes together.
     * This must be called on all nodes in the group before computePartials.
     *
     * @param nodes Vector of all nodes that should be normalized together
     */
    static void apply_softmax(std::vector<Node*>& nodes) {
        if (nodes.empty()) return;

        // Collect values (logits) before softmax
        std::vector<double> logits;
        logits.reserve(nodes.size());
        for (Node* node : nodes) {
            logits.push_back(node->value);
        }

        // Compute softmax
        std::vector<double> softmax_outputs = softmax(logits);

        // Update each node with its softmax output and metadata
        for (size_t i = 0; i < nodes.size(); ++i) {
            nodes.at(i)->value = softmax_outputs.at(i);
            nodes.at(i)->isSoftmaxApplied = true;
            nodes.at(i)->softmaxGroup = nodes;  // Store reference to the group
            nodes.at(i)->softmaxOutputs = softmax_outputs;  // Cache outputs
            nodes.at(i)->softmaxIndex = static_cast<int>(i);  // Store this node's index
        }
    }

    void computePartials() {
        // 0) Reset flags/gradients so repeated calls work predictably
        std::stack<Node *> st;
        st.push(this);
        while (!st.empty()) {
            Node *n = st.top();
            st.pop();
            if(n->isVisited) {
                n->isVisited = false;
                if (n->var) n->gradient = 0.0;
                if (n->right != nullptr) st.push(n->right);
                if (n->left != nullptr) st.push(n->left);
            }
        }

        // 1) Reverse-mode pass using an explicit stack
        std::stack<std::tuple<Node *, double> > nodeStack;
        nodeStack.emplace(this, 1.0);

        while (!nodeStack.empty()) {
            Node* tempNode = std::get<0>(nodeStack.top());
            double tempSeed = std::get<1>(nodeStack.top());
            nodeStack.pop();

            // Handle sigmoid activation
            if (tempNode->isSigmoidApplied == true) {
                tempSeed *= sigmoid_derivative(tempNode->get_value());
            }

            // Handle softmax activation
            if (tempNode->isSoftmaxApplied == true) {
                // Softmax backward pass requires all nodes in the group
                // We need to distribute gradients according to the Jacobian

                if (!tempNode->softmaxGroup.empty() && !tempNode->softmaxOutputs.empty()) {
                    // Compute the contribution of this gradient seed to all logits
                    // using: dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
                    const int idx = tempNode->softmaxIndex;
                    const std::vector<double>& y = tempNode->softmaxOutputs;
                    // This gradient is for output idx, so we create grad_output vector
                    std::vector<double> grad_output(y.size(), 0.0);
                    grad_output.at(idx) = tempSeed;
                    // Compute gradient w.r.t. all logits
                    std::vector<double> grad_logits = softmax_backward(y, grad_output);
                    // Distribute gradients to all nodes in the softmax group
                    // Note: We need to accumulate these at the pre-softmax values
                    // So we modify tempSeed to represent the gradient at the input
                    tempSeed *= grad_logits.at(idx); //TODO: Check the mathematical validity of this
                }
            }

            // Accumulate gradient if this node is a leaf variable
            if (tempNode->var) {
                tempNode->gradient += tempSeed;
            }

            // Push children exactly once per node
            if (!tempNode->isVisited) {
                Node *L = tempNode->left;
                Node *R = tempNode->right;

                switch (tempNode->operation) {
                    case '+':
                        if (R && R != nullptr) nodeStack.emplace(R, tempSeed);
                        if (L && L != nullptr) nodeStack.emplace(L, tempSeed);
                        break;
                    case '*':
                        if (R && R != nullptr) nodeStack.emplace(R, (L->get_value() * tempSeed));
                        if (L && L != nullptr) nodeStack.emplace(L, (R->get_value() * tempSeed));
                        break;
                    case '/':
                        // y = L / R  ->  dy/dL = 1/R, dy/dR = -L/(R^2)
                        if (R && R != nullptr) nodeStack.emplace(R, (-L->get_value() / (R->get_value() * R->get_value())) * tempSeed);
                        if (L && L != nullptr) nodeStack.emplace(L, (1.0 / R->get_value()) * tempSeed);
                        break;
                    default:
                        break;
                }

                tempNode->isVisited = true;
            }
        }
    }
};

#endif //EXPRESSIONNODE_H
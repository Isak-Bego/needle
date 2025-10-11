#ifndef EXPRESSIONNODE_H
#define EXPRESSIONNODE_H

#include "./activation-functions/sigmoid.h"
#include <stack>
#include <tuple>

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
    std::vector<Node *> ownedNodes; // Track nodes we created

public:
    explicit Node(double value, bool isVariable) {
        this->value = value;
        this->var = isVariable;
    }

    ~Node() {
        for (Node *n: ownedNodes) {
            delete n;
        }
    }

    Node(double value, Node *left, Node *right, char operation) {
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

    Node *operator+(Node &right) {
        Node *result = new Node(this->value + right.value, this, &right, '+');
        ownedNodes.push_back(result);
        return result;
    }

    Node *operator*(Node &right) {
        Node *result = new Node(this->value * right.value, this, &right, '*');
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

    Node *get_right() {
        return right;
    }

    double get_gradient() {
        return gradient;
    }

    char get_operation() {
        return operation;
    }

    void apply_sigmoid() {
        this->value = sigmoid(this->value);
        this->isSigmoidApplied = true;
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
                if (n->var) n->gradient = 0.0;
                if (n->right) st.push(n->right);
                if (n->left) st.push(n->left);
            }
        }

        // 1) Reverse-mode pass using an explicit stack
        std::stack<std::tuple<Node *, double> > nodeStack;
        nodeStack.emplace(this, 1.0);

        while (!nodeStack.empty()) {
            auto tempNode = std::get<0>(nodeStack.top());
            auto tempSeed = std::get<1>(nodeStack.top());
            nodeStack.pop();

            if (tempNode->isSigmoidApplied == true) {
                tempSeed *= sigmoid_derivative(tempNode->get_value());
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
                    default:
                        break;
                }

                tempNode->isVisited = true;
            }
        }
    }
};

#endif //EXPRESSIONNODE_H

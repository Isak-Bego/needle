#ifndef EXPRESSIONNODE_H
#define EXPRESSIONNODE_H

#include "utils/activation-functions/sigmoid.h"
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

    void computePartials() {
        // is big
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
            const auto tempNode = std::get<0>(nodeStack.top());
            auto tempSeed = std::get<1>(nodeStack.top());
            nodeStack.pop();

            if (tempNode->isSigmoidApplied == true) {
                tempSeed *= sigmoid_derivative(tempNode->get_value());
            }

            // Accumulate gradient if this node is a leaf variable
            if (tempNode->var) {
                tempNode->gradient += tempSeed;
                // std::cout<<"Temp seed: "<<tempSeed<<std::endl;
                // std::cout<<"TempNode gradient"<<tempNode->gradient<<std::endl;
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
                        if (R && R != nullptr) {
                            // std::cout<<"Temp seed update: "<<tempSeed<<" * "<<L->get_value()<<" = "<<(L->get_value() * tempSeed)<<std::endl;
                            nodeStack.emplace(R, (L->get_value() * tempSeed));
                        }
                        if (L && L != nullptr) {
                            // std::cout<<"Temp seed update: "<<tempSeed<<" * "<<R->get_value()<<" = "<<(R->get_value() * tempSeed)<<std::endl;
                            nodeStack.emplace(L, (R->get_value() * tempSeed));
                        }
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

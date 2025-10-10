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
    std::vector<Node*> ownedNodes; // Track nodes we created

public:
    explicit Node(float value, bool isVariable) {
        this->value = value;
        this->var = isVariable;
    }

    ~Node() {
        for (Node* n : ownedNodes) {
            delete n;
        }
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

    Node* operator+(Node& right) {
        Node* result = new Node(this->value + right.value, this, &right, '+');
        ownedNodes.push_back(result);
        return result;
    }

    Node* operator*(Node& right) {
        Node* result = new Node(this->value * right.value, this, &right, '*');
        ownedNodes.push_back(result);
        return result;
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
        std::stack<std::tuple<Node *, float> > nodeStack;
        nodeStack.emplace(this, 1.0f);

        while (!nodeStack.empty()) {
            auto tempNode = std::get<0>(nodeStack.top());
            auto tempSeed = std::get<1>(nodeStack.top());
            nodeStack.pop();

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
                        if (R && R != nullptr) nodeStack.emplace(R, (L->value * tempSeed));
                        if (L && L != nullptr) nodeStack.emplace(L, (R->value * tempSeed));
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
    Node b = Node(3.9, true);
    Node c = Node(6.0, true);
    Node d = Node(4.6, true);

    Node* e = b + a;
    e = (*e) * d;
    e = (*e) + c;


    e->computePartials();

    std::cout << a.get_gradient() << std::endl;
    std::cout << b.get_gradient() << std::endl;
    std::cout << c.get_gradient() << std::endl;
    std::cout<< d.get_gradient() << std::endl;

    return 0;
}

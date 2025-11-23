#ifndef NODE_H
#define NODE_H
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <unordered_set>
#include <iostream>

/**
* An auto-differentiation engine that is based on an expression tree, where the gradients flow from
* last element in a topological sorted tree, all the way to its root elements. This class is the core of the project,
* because it provides the necessary tools for both forward-passing and backward-passing, the two fundamental iterations
* of Machine Learning.
*
* The inspiration for the implementation of this class is taken by the Wikipedia article about auto-differentiation:
* https://en.wikipedia.org/wiki/Automatic_differentiation
*/
class Node {
public:
    // stores a single scalar value and its gradient
    double data;
    double grad;

    // internal variables used for autograd graph construction
    std::function<void()> backwardProp;
    std::vector<Node *> previousNodes; // parents in the computation graph
    std::string operation; // the operation that produced this node could be arithmetic or another function

    /**
     * @brief Constructs the node object given the data, the children that derived it and the operation that was used
     * to compute the data.
     *
     * @param data - The scalar data that the node encapsulates
     * @param children - The immediate nodes that were involved in the computation of the current node
     * @param op - The operation which the children underwent to produce the parent node
     */
    explicit Node(double data, const std::vector<Node *> &children = {}, const std::string &op = "")
        : data(data), grad(0.0), backwardProp([] {
        }), previousNodes(children), operation(op) {
    }


    /**
     * @brief Raises the data of the calling node object to the power specified by the parameter {other}
     *
     * @param other - The exponent of the power function
     * @return
     */
    Node *pow(double other) {
        auto self = this;
        const double out_data = std::pow(self->data, other);
        auto out = new Node(out_data, {self}, "**" + std::to_string(other));

        out->backwardProp = [self, out, other]() {
            self->grad += (other * std::pow(self->data, other - 1)) * out->grad;
        };

        return out;
    }

    /**
     *
     * @param x - The input of the log function
     * @param epsilon - Minimum input to the log function
     * @return - Returns a new node with data = log(x)
     */
    static Node *logNode(Node *x, double epsilon = 1e-7) {
        auto self = x;
        // We clamp the data so that we do not run into issue that are caused by the calculation of log(0)
        const double clampedData = std::max(self->data, epsilon);
        const double outData = std::log(clampedData);
        auto out = new Node(outData, {self}, "log");

        out->backwardProp = [self, out, epsilon]() {
            const double clamped = std::max(self->data, epsilon);
            self->grad += (1.0 / clamped) * out->grad;
        };

        return out;
    }

    /**
     * @brief Topologically sorts all the nodes in the expression graph where the calling node belongs to ,and
     * then it runs back-propagation from the last node in the topologically sorted list to make sure the chain
     * rule is followed rigorously.
     *
     */
    void backward() {
        // topological order all the children in the graph
        std::vector<Node *> topo;
        std::unordered_set<Node *> visited;

        std::function<void(Node *)> build_topo = [&](Node *v) {
            if (!v || visited.count(v)) return;
            visited.insert(v);
            for (Node *child: v->previousNodes) {
                build_topo(child);
            }
            topo.push_back(v);
        };

        build_topo(this);

        // go one variable at a time and apply the chain rule to get its gradient
        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backwardProp();
        }
    }
};

/*
 * The following is a list of operator overloading functions that contribute in the computation of the
 * arithmetic expressions between (Node) objects. They play a crucial role in building the expression tree
 * all while abstracting that complexity from the user, who essentially just carries arithmetic as we would
 * regularly do.
 */

inline Node *operator+(Node &a, Node &b) {
    auto out = new Node(a.data + b.data, {&a, &b}, "+");

    Node *pa = &a;
    Node *pb = &b;

    out->backwardProp = [pa, pb, out]() {
        pa->grad += out->grad;
        pb->grad += out->grad;
    };

    return out;
}


inline Node *operator+(Node &a, const double b) {
    auto pb = new Node(b);
    auto out = new Node(a.data + pb->data, {&a, pb}, "+");

    Node *pa = &a;
    out->backwardProp = [pa, pb, out]() {
        pa->grad += out->grad;
        pb->grad += out->grad;
    };

    return out;
}

inline Node *operator+(const double a, Node &b) {
    auto pa = new Node(a);
    auto out = new Node(pa->data + b.data, {pa, &b}, "+");

    Node *pb = &b;
    out->backwardProp = [pa, pb, out]() {
        pa->grad += out->grad;
        pb->grad += out->grad;
    };

    return out;
}


inline Node *operator*(Node &a, Node &b) {
    auto out = new Node(a.data * b.data, {&a, &b}, "*");

    Node *pa = &a;
    Node *pb = &b;
    out->backwardProp = [pa, pb, out]() {
        pa->grad += pb->data * out->grad;
        pb->grad += pa->data * out->grad;
    };

    return out;
}

inline Node *operator*(Node &a, const double b) {
    auto pb = new Node(b);
    auto out = new Node(a.data * pb->data, {&a, pb}, "*");

    Node *pa = &a;
    out->backwardProp = [pa, pb, out]() {
        pa->grad += pb->data * out->grad;
        pb->grad += pa->data * out->grad;
    };

    return out;
}

inline Node *operator*(const double a, Node &b) {
    auto pa = new Node(a);
    auto out = new Node(pa->data * b.data, {pa, &b}, "*");

    Node *pb = &b;
    out->backwardProp = [pa, pb, out]() {
        pa->grad += pb->data * out->grad;
        pb->grad += pa->data * out->grad;
    };

    return out;
}


inline Node *operator-(Node &a) {
    return a * -1.0;
}


inline Node *operator/(Node &a, Node &b) {
    auto out = new Node(a.data / b.data, {&a, &b}, "/");

    Node *pa = &a;
    Node *pb = &b;

    out->backwardProp = [pa, pb, out]() {
        pa->grad += (1.0 / pb->data) * out->grad;

        const double b2 = pb->data * pb->data;
        pb->grad += (-pa->data / b2) * out->grad;
    };

    return out;
}


inline Node *operator/(Node &a, double b) {
    auto out = new Node(a.data / b, {&a}, "/");

    Node *pa = &a;

    out->backwardProp = [pa, b, out]() {
        // dz/da = 1 / b
        pa->grad += (1.0 / b) * out->grad;
        // b is a plain double constant, no grad
    };

    return out;
}

inline Node *operator/(double a, Node &b) {
    auto out = new Node(a / b.data, {&b}, "/");

    Node *pb = &b;

    out->backwardProp = [pb, a, out]() {
        // dz/db = -a / b^2
        const double b2 = pb->data * pb->data;
        pb->grad += (-a / b2) * out->grad;
        // a is a plain constant, no grad
    };

    return out;
}


inline std::ostream &operator<<(std::ostream &os, const Node &n) {
    os << "Node(data=" << n.data << ", grad=" << n.grad << ")";
    return os;
}

#endif //NODE_H

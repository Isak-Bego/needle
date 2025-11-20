#ifndef SGD_H
#define SGD_H
#include <vector>
#include <autoGradEngine/node.h>

class SGD {
    double learning_rate;

public:
    explicit SGD(const double lr = 0.01) : learning_rate(lr) {
    }

    void step(std::vector<Node *> &parameters) const {
        for (Node *param: parameters) {
            if (param) {
                // Update: param = param - learning_rate * gradient
                param->data -= learning_rate * param->grad;
            }
        }
    }

    void set_learning_rate(const double lr) {
        learning_rate = lr;
    }

    double get_learning_rate() const {
        return learning_rate;
    }
};

#endif //SGD_H

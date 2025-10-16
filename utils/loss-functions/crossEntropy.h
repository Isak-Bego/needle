#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H
#include "utils/expressionNode.h"
#include <cmath>

class CrossEntropy : public Node{
    double crossEntropyError = 0.0;
public:
    explicit CrossEntropy() : Node(0.0, nullptr, nullptr, 'f') {}

    void calculateError() {
        this->crossEntropyError = -log(this->get_value());
    }

    double computeActivationPartial() override {
        return -1.0/this->get_value();
    }
};

#endif //CROSSENTROPY_H

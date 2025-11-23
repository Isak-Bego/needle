#ifndef SGD_H
#define SGD_H
#include <vector>
#include <autoGradEngine/node.h>

/**
 *  This function defines the methods that optimize the parameters of a model based on their partial derivatives to
 *  the loss function. This way we aim to minimize loss and gain accuracy so that we can put our model to use.
 */
class SGD {
    double learningRate;

public:
    explicit SGD(const double lr = 0.01) : learningRate(lr) {
    }

    /**
     * @brief This function updates every parameter of the model by adding the negated gradient scaled by the
     * learning rate. Its aim is to decrease the loss function at every step.
     *
     * @param parameters - The parameters of a model
     */
    void step(std::vector<Node *> &parameters) const {
        for (Node *param: parameters) {
            if (param) {
                // Update: param = param - learning_rate * gradient
                param->data -= learningRate * param->grad;
            }
        }
    }

    // Getters and setters for the learning rate
    void setLearningRate(const double lr) {
        learningRate = lr;
    }

    double getLearningRate() const {
        return learningRate;
    }
};

#endif //SGD_H

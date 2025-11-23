#ifndef BINARYCROSSENTROPY_H
#define BINARYCROSSENTROPY_H

#include <autoGradEngine/node.h>

/**
 *  The binary cross-entropy loss function computes the loss for the binary classifier given the certainty that the model
 *  has in the correct target class. By minimizing the value of this function, we train the model to make the distinction
 *  between two classes of objects.
 */
class BinaryCrossEntropyLoss {
public:
    /**
     * @brief This function is used to produce the loss value for a given prediction to a certain target of interest.
     *
     * @param prediction - The certainty of the model that the correct output is @p target
     * @param target - The desired output
     * @param epsilon - The minimum value of the input passed to the log function so that we prevent log(0)
     * @return - The value of the binary classifier loss
     */
    static Node *compute(Node *prediction, double const target, double const epsilon = 1e-7) {
        const auto predClamped = prediction;
        const auto logPred = Node::logNode(predClamped, epsilon);
        const auto term1 = (*logPred) * (-target);

        const auto oneMinusPred = *((*predClamped) * (-1.0)) + 1.0;
        const auto logOneMinusPred = Node::logNode(oneMinusPred, epsilon);
        const auto term2 = (*logOneMinusPred) * (-(1.0 - target));

        const auto loss = (*term1) + (*term2);
        return loss;
    }
};


#endif //BINARYCROSSENTROPY_H

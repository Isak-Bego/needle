#ifndef CATEGORICALCROSSENTROPY_H
#define CATEGORICALCROSSENTROPY_H

#include <autoGradEngine/node.h>
#include <vector>

/**
 *  The categorical cross-entropy loss function computes the loss based on the probability distribution all possible targets.
 *  It computes how certain our model is in assigning the biggest probability to the target class, while assigning probabilities
 *  close to zero for the other classes. By minimizing this function, our model is able to classify inputs among several
 *  classes.
 */
class CategoricalCrossEntropyLoss {
public:
    /**
     * @brief This function is used to produce the loss value for a given prediction to a certain target of interest.
     *
     * @param predictions - The probability distribution that is created by the SoftMax function.
     * @param target - The desired output
     * @param epsilon - The minimum value of the input passed to the log function so that we prevent log(0)
     * @return - The value of the multi-class classifier loss
     */
    static Node *compute(const std::vector<Node *> &predictions, int target,
                         const double epsilon = 1e-7) {
        if (predictions.empty()) {
            std::cout << "Predictions vector cannot be empty";
        }
        if (target < 0 || target >= static_cast<int>(predictions.size())) {
            std::cout << "Target class out of range";
        }

        auto logProb = Node::logNode(predictions.at(target), epsilon);
        const auto loss = (*logProb) * (-1.0);

        return loss;
    }
};

#endif //CATEGORICALCROSSENTROPY_H

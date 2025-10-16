#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H
#include <utils/node.h>
#include <nn_components/layer.h>
#include <nn_components/neuron.h>
#include <utils/helperFunctions.h>
#include <cmath>

class CrossEntropyNode final : public Node {
    double predictedProbability = 0.0;

public:
    explicit CrossEntropyNode() : Node(0.0, nullptr, nullptr, 'f') {
    }

    void set_value(const double &tempVal) override {
        this->predictedProbability = tempVal;
        Node::set_value(tempVal);
        this->calculateError();
    }

    void calculateError() {
        Node::set_value(-log(this->get_value()));
    }

    double computeActivationPartial() override {
        return -1.0 / this->predictedProbability;
    }
};

class CrossEntropyLayer final : public Layer {
    double expectedOutput = 0.0;
    std::vector<double> distinctClassificationClasses;

public:
    explicit CrossEntropyLayer(const int numberOfNeurons, const std::vector<double> &distinctClassificationClasses,
                               const double expectedOutput = 0.0) : Layer(numberOfNeurons) {
        this->expectedOutput = expectedOutput;
        this->distinctClassificationClasses = distinctClassificationClasses;

        std::vector<Neuron> errorLayerNeurons;
        errorLayerNeurons.reserve(1);
        auto *error = new CrossEntropyNode();
        this->getNeurons().emplace_back();
        errorLayerNeurons.emplace_back();
        errorLayerNeurons.back().setActivationNode(error);
        this->setNeurons(errorLayerNeurons);
    }

    CrossEntropyLayer(const int numberOfNeurons, Layer *previousLayer) : Layer(numberOfNeurons, previousLayer) {
    }

    void forwardPass() override {
        const Neuron prediction = (this->getPreviousLayer()->getNeurons().at(
            helper::find(distinctClassificationClasses, expectedOutput)));
        this->getNeurons().front().getActivation()->set_value(prediction.getActivation()->get_value());
        this->getNeurons().front().getActivation()->set_left(prediction.getActivation());
    }

    void print() override {
        std::cout << std::endl << "---------Error Layer Begin------------" << std::endl << std::endl;
        std::cout << "Cross Entropy Loss: " << this->getNeurons().front().getActivation()->get_value() << std::endl <<
                std::endl;
        std::cout << "-----------Error Layer End-------------" << std::endl;
    }

    void setExpectedOutput(const double expectedOutput) { this->expectedOutput = expectedOutput; }
};

#endif //CROSSENTROPY_H

#ifndef SGDOPTIMIZER_H
#define SGDOPTIMIZER_H
#include <nn_components/network.h>
void sgdOptimizer(Network& net) {
    auto layers = net.getLayers();
    for (Layer* layer : layers) {
        auto neurons = layer->getNeurons();
        for (Neuron& neuron : neurons) {
            auto& weights = neuron.getWeights();
            for (Node& weight : weights) {
                weight.set_value(weight.get_value() - 0.15*weight.get_gradient());
            }
        }
    }
}
#endif //SGDOPTIMIZER_H

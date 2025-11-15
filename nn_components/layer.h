#ifndef LAYER_H
#define LAYER_H
#include <nn_components/module.h>
#include <nn_components/neuron.h>

class Layer : public Module {
    std::vector<Neuron> neurons;

public:
    Layer(int nin, int nout, bool nonlin = true) {
        neurons.reserve(nout);
        for (int i = 0; i < nout; ++i) {
            neurons.emplace_back(nin, nonlin);
        }
    }

    std::vector<Node*> operator()(const std::vector<Node*>& x) {
        std::vector<Node*> out;
        out.reserve(neurons.size());
        for (auto& n : neurons) {
            out.push_back(n(x));
        }
        return out;
    }

    std::vector<Node*> parameters() override {
        std::vector<Node*> params;
        for (auto& n : neurons) {
            auto np = n.parameters();
            params.insert(params.end(), np.begin(), np.end());
        }
        return params;
    }

    std::string repr() const {
        std::string s = "Layer of [";
        for (size_t i = 0; i < neurons.size(); ++i) {
            s += neurons.at(i).repr();
            if (i + 1 < neurons.size()) s += ", ";
        }
        return s + "]";
    }
};

inline std::ostream& operator<<(std::ostream& os, const Layer& l) {
    return os << l.repr();
}

#endif //LAYER_H

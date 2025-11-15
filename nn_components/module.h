#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <utils/node.h>

class Module {
public:
    virtual ~Module() = default;

    virtual std::vector<Node*> parameters() {
        return {};
    }

    void zero_grad() {
        for (Node* p : parameters()) {
            if (p) p->grad = 0.0;
        }
    }
};


#endif //MODULE_H

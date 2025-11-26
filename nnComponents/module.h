#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <autoGradEngine/node.h>

/**
 * Module is a virtual class that serves as an interface for the Network class, enforcing the implementation of
 * the parameters() and clear_gradients() methods, which are important in the ability to access the parameters and
 * reset their gradients.
 */
class Module {
public:
    virtual ~Module() = default;

    virtual std::vector<Node *> parameters() {
        return {};
    }

    // Clears all the gradients of the parameters so that they are ready for the next backward pass
    void clearGradients() {
        for (Node *p: parameters()) {
            if (p) p->grad = 0.0;
        }
    }
};


#endif //MODULE_H

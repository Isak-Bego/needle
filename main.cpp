#include <iostream>
#include <utils/node.h>
#include <nnComponents/network.h>

int main() {
    Network model(2, {16, 16, 2});
    std::cout << model << "\n";

    Node x0(1.0), x1(-2.0);
    std::vector<Node*> x = { &x0, &x1 };

    auto out = model(x);

    std::cout << "Output: " << out.at(0)->data << "\n";

    out.at(0)->backward();
    out.at(1)->backward();
    for(Node* n : model.parameters()) {
       std::cout << n->data << "\n";
    }
    std::cout << "grad x0 = " << x0.grad << "\n";
    std::cout << "grad x1 = " << x1.grad << "\n";
}


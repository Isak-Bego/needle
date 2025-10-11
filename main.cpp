#include <iostream>
#include "utils/expressionNode.h"

int main() {
    Node a = Node(1, false);
    Node b = Node(2, true);
    Node c = Node(3, true);
    Node d = Node(4, false);

    Node* e = b + c;

    e->computePartials();

    std::cout<<"My results: "<<std::endl;
    std::cout << a.get_gradient() << std::endl;
    std::cout << b.get_gradient() << std::endl;
    std::cout << c.get_gradient() << std::endl;
    std::cout<< d.get_gradient() << std::endl;

    return 0;
}

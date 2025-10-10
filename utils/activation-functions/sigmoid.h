//
// Created by Isak Bego on 27/9/25.
//

#ifndef SIGMOID_H
#define SIGMOID_H
#include <cmath>

double sigmoid(double x){
  return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double sigmoid_output) {
  return sigmoid_output * (1.0 - sigmoid_output);
}

#endif //SIGMOID_H

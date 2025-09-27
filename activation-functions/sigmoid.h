//
// Created by Isak Bego on 27/9/25.
//

#ifndef SIGMOID_H
#define SIGMOID_H
#include <cmath>

float sigmoid(float x){
  return 1 / (1 + exp(-x));
}

#endif //SIGMOID_H

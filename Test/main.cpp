#include <iostream>
#include <cmath>

#include "../Library/Neural/Perceptron.h"

using namespace std;

int main() {
    Perceptron<float> p({22, 50}, {-1071.15, -2435.07});
    p.setBias(1);
    p.setBiasWeight(-47.7113);
    p.setLearningRate(0.5);
    p.setTarget(72.0);
    p.display();
    p.setActivationFunction("linear");
    
    p.train(true);
    p.display();

    return 0;
}
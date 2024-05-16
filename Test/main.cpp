#include <iostream>
#include <cmath>

#include "../Library/Neural/Perceptron.h"

using namespace std;

int main() {
    Perceptron<float> p({0, 0}, {0.1, 0.1});
    p.setBias(1);
    p.setBiasWeight(0.1);
    p.setLearningRate(0.5);
    p.setTarget(0);
    p.setActivationFunction("step");
    p.setAccuracy(0.001);
    p.feedForward();
    p.display();

    p.train(true);
    p.display();
    return 0;
}
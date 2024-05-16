#include <iostream>
#include <cmath>

#include "../Library/Neural/Perceptron.h"

using namespace std;

int main() {
    Perceptron<float> p({1, 1}, {-1, 0});
    p.setBias(0.5);
    p.setLearningRate(0.5);
    p.setTarget(1);
    p.feedForward();
    p.display();
    p.setActivationFunction("step");
    p.train();
    p.display();

    return 0;
}
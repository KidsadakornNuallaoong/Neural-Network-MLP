#include <iostream>
#include <cmath>

#include "../Library/Neural/Perceptron.h"

using namespace std;

int main() {
    Perceptron<float> p({1, 1}, {-0.5, 0.5});
    p.setBias(1);
    p.setLearningRate(0.5);
    p.setTarget(0);
    p.feedForward();
    p.display();
    p.train();
    p.display();

    return 0;
}
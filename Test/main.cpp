#include <iostream>
#include <cmath>

#include "../Library/Neural/Perceptron.h"

using namespace std;

int main() {
    // * Model train
    Perceptron<double> p({22, 50}, {0.1, 0.1});
    p.setBias(1);
    p.setBiasWeight(0.1);
    p.setLearningRate(0.0001);
    p.setTarget(75.995);
    p.setActivationFunction("Linear");
    p.setAccuracy(0.001);
    p.feedForward();
    p.display();

    p.train(true);
    p.display();

    // * Model test
    Perceptron<double> p2({22, 50}, p.getWeights(), p.getBias(), p.getBiasWeight(), p.getLearningRate(), p.getAccuracy(), p.getTarget());
    p2.setActivationFunction("Linear");
    p2.feedForward();
    p2.display();

    return 0;
}
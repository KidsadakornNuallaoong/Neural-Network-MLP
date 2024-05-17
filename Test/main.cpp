#include <iostream>
#include <cmath>

#include "../Library/Neural/Perceptron.h"

using namespace std;

int main() {
    // // * Model train
    cout << "\033[1;31m" << "=======>> Model train <<=======" << "\033[0m" << endl;
    Perceptron<double> p({22, 50}, {0.1, 0.1});
    p.setBias(1);
    p.setBiasWeight(0.1);
    p.setLearningRate(0.0001);
    p.setTarget(75.995);
    p.setActivationFunction("Linear");
    p.setAccuracy(1e-5);
    p.feedForward();
    p.display();

    p.train(true);
    p.display();

    // * Model test
    cout << "\033[1;31m" << "=======>> Model test <<=======" << "\033[0m" << endl;
    // Perceptron<double> p2({22, 50}, p.getWeights(), p.getBias(), p.getBiasWeight(), p.getLearningRate(), p.getAccuracy(), p.getTarget());
    // p2.setActivationFunction("Linear");
    // p2.feedForward();
    // p2.display();

    Perceptron<double> p2({22, 50});
    p2.copyPerceptron(p);
    p2.setActivationFunction("Linear");
    p2.feedForward();
    p2.display();

    // Perceptron<float> ptest_0({0.35, 0.9}, {0.0991, 0.7976});
    // ptest_0.setBias(0);
    // ptest_0.setBiasWeight(0);
    // ptest_0.setActivationFunction("Sigmoid");
    // ptest_0.feedForward();
    // ptest_0.display();

    // Perceptron<float> ptest_1({0.35, 0.9}, {0.3971, 0.5926});
    // ptest_1.setBias(0);
    // ptest_1.setBiasWeight(0);
    // ptest_1.setActivationFunction("Sigmoid");
    // ptest_1.feedForward();
    // ptest_1.display();

    // Perceptron<float> ptest_2({ptest_0.getOutput(), ptest_1.getOutput()}, {0.2724, 0.8731});
    // ptest_2.setBias(0);
    // ptest_2.setBiasWeight(0);
    // ptest_2.setActivationFunction("Sigmoid");
    // ptest_2.feedForward();
    // ptest_2.display();

    return 0;
}
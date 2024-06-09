#include <iostream>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>

#include "../Library/Perceptron/Perceptron.h"
#include "../Library/Neural/Neural.h"
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

int main() {
    Perceptron<float> p1;
    vector<float> inputs = {1, -2};
    vector<float> weights = {-0.1, -0.1};
    p1.setInputs(inputs);
    p1.setWeights(weights);
    p1.setBias(0.5);
    p1.setLearningRate(0.01);
    p1.setTarget(-1);
    p1.setAccuracy(0.01);
    p1.train(1000, 1e-4);
    p1.setActivation("Linear");
    p1.feedForward();
    p1.display();
    cout << endl;

    p1.setActivation("ReLU");
    p1.feedForward();
    p1.display();
    cout << endl;

    p1.setActivation("LeakyReLU");
    p1.feedForward();
    p1.display();
    cout << endl;
}
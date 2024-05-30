#include <iostream>
#include <cmath>
#include <vector>

#include "../Library/Perceptron/Perceptron.h"
#include "../Library/Neural/Neural.h"
#include "matplotlibcpp.h"

using namespace std;

namespace plt = matplotlibcpp;

int main() {
    Perceptron<double> p;
    p.setInputs({1});
    p.setWeights({0.1});
    p.setBias(0.1);
    p.setActivation("Linear");
    p.setTarget(2);
    p.setLearningRate(0.001);
    p.feedForward();
    p.display();

    cout << endl;

    plt::ion();

    vector<double> x_data, y_data, input, output;
    for (int i = 0; i < 1000; i++) {
        p.backpropagate();
        x_data.push_back(i);
        y_data.push_back(p.MSE());
        plt::cla();
        plt::plot(x_data, y_data, "r-");
        plt::pause(0.01);
    }
    p.display();
    cout << endl;

    Perceptron<double> p2;
    p2.setInputs({3});
    p2.copyEnv(&p);
    p2.setTarget(4);
    p2.feedForward();
    p2.display();

    return 0;
}
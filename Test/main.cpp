#include <iostream>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>

#include "../Library/Perceptron/Perceptron.h"
#include "../Library/Neural/Neural.h"
// #include "matplotlibcpp.h"

using namespace std;
// namespace plt = matplotlibcpp;

/*
int main() {
    Perceptron<float> p1, p2, p3;

    float target = 0.5;

    vector<float> inputs = {0.35, 0.9};
    vector<float> weights_1 = {0.1, 0.8};

    p1.setInputs(inputs);
    p1.setWeights(weights_1);
    p1.setBias(0);
    p1.setLearningRate(0.01);
    p1.setTarget(target);
    p1.setAccuracy(0.01);

    cout << "\033[1;31mPerceptron node 1\033[0m" << endl << endl;

    p1.setActivation("Linear");
    p1.feedForward();
    p1.display();
    cout << endl;

    p1.setActivation("Sigmoid");
    p1.feedForward();
    p1.display();
    cout << endl;

    cout << "\033[1;31mPerceptron node 2\033[0m" << endl << endl;

    vector<float> weights_2 = {0.4, 0.6};

    p2.setInputs(inputs);
    p2.setWeights(weights_2);
    p2.setBias(0);
    p2.setLearningRate(0.01);
    p2.setTarget(target);
    p2.setAccuracy(0.01);

    p2.setActivation("Linear");
    p2.feedForward();
    p2.display();
    cout << endl;

    p2.setActivation("Sigmoid");
    p2.feedForward();
    p2.display();
    cout << endl;

    cout << "\033[1;31mPerceptron node 3\033[0m" << endl << endl;

    vector<float> weights_3 = {0.3, 0.9};

    p3.setInputs({p1.getOutput(), p2.getOutput()});
    p3.setWeights(weights_3);
    p3.setBias(0);
    p3.setLearningRate(0.01);
    p3.setTarget(target);
    p3.setAccuracy(0.01);

    p3.setActivation("Linear");
    p3.feedForward();
    p3.display();
    cout << endl;

    p3.setActivation("Sigmoid");
    p3.feedForward();
    p3.display();
    cout << endl;

    return 0;
}
*/

int main() {
    Perceptron<float> p1[2];
    vector<float> inputs = {1, 1, 0, 1};
    vector<float> weight[2] = {{0.3, -0.2, 0.2, 0.1}, {0.3, 0.4, -0.3, 0.4}};
    vector<float> weight2 = {-0.3, 0.2};
    vector<float> bias = {0.2, 0.1, -0.3};

    vector<int> target = {1, 1, 0, 1};

    cout << "\033[1;31mPerceptron node 1\033[0m" << endl << endl;

    p1[0].setInputs(inputs);
    p1[0].setWeights(weight[0]);
    p1[0].setBias(bias[0]);
    p1[0].setActivation("Sigmoid");
    p1[0].feedForward();
    p1[0].display();
    cout << endl;

    cout << "\033[1;31mPerceptron node 2\033[0m" << endl << endl;

    p1[1].setInputs(inputs);
    p1[1].setWeights(weight[1]);
    p1[1].setBias(bias[1]);
    p1[1].setActivation("Sigmoid");
    p1[1].feedForward();
    p1[1].display();
    cout << endl;

    cout << "\033[1;31mPerceptron node 3\033[0m" << endl << endl;

    Perceptron<float> p2;
    p2.setInputs({p1[0].getOutput(), p1[1].getOutput()});
    p2.setWeights(weight2);
    p2.setBias(bias[2]);
    p2.setActivation("Sigmoid");
    p2.setTarget(target[0]);
    p2.feedForward();
    p2.display();
    cout << endl;
}
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

    cout << "\033[1;31mPerceptron node 1\033[0m" << endl;

    p1.setActivation("Linear");
    p1.feedForward();
    p1.display();
    cout << endl;

    p1.setActivation("Sigmoid");
    p1.feedForward();
    p1.display();
    cout << endl;

    cout << "\033[1;31mPerceptron node 2\033[0m" << endl;

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

    cout << "\033[1;31mPerceptron node 3\033[0m" << endl;

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

    // // * plot input and loss of model
    // // plt::plot(p1.getInputs(), p1.getWeights(), "r-");
    // // plt::scatter_colored(dat_a, dat_b, w, 50);
    // // plt::plot(p1.getInputs(), p1.getWeights(), "r-");
    // // vector<float> loss;
    // // for (int i = 0; i < p1.getInputs().size(); i++) {
    // //     p1.setInputs({p1.getInputs()[i]});
    // //     p1.feedForward();
    // //     loss.push_back(p1.getOutput());
    // // }
    // // plt::scatter(p1.getInputs(), loss, 50);
    // // plt::xlabel("Input");
    // // plt::ylabel("Loss");
    // // plt::show();

    // Perceptron<float> p2(p1);
    // p2.setInputs({1, 2});
    // p2.setWeights({0.1, 0.1});
    // p2.feedForward();
    // p2.display();
}
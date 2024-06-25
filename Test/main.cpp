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

// * plot input and loss of model
    // plt::plot(p1.getInputs(), p1.getWeights(), "r-");
    // plt::scatter_colored(dat_a, dat_b, w, 50);
    // plt::plot(p1.getInputs(), p1.getWeights(), "r-");
    // vector<float> loss;
    // for (int i = 0; i < p1.getInputs().size(); i++) {
    //     p1.setInputs({p1.getInputs()[i]});
    //     p1.feedForward();
    //     loss.push_back(p1.getOutput());
    // }
    // plt::scatter(p1.getInputs(), loss, 50);
    // plt::xlabel("Input");
    // plt::ylabel("Loss");
    // plt::show();

    Perceptron<float> p2(p1);
    p2.setInputs({1, 2});
    p2.setWeights({0.1, 0.1});
    p2.feedForward();
    p2.display();
}
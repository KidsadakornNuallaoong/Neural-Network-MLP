#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// #include "../Library/Perceptron/Perceptron.hpp"
#include "../Library/MLP/MLP.hpp"

int main() {

    // * Test Multi-Layer Perceptron
    vector<int> layersSize = {2, 8, 8, 1};
    MultiLayerPerceptron<double> mlp(layersSize);

    // * XOR Problem
    // vector<vector<double>> inputs = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    // vector<vector<double>> targets = {{3}, {5}, {7}, {9}};
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};

    mlp.setActivation({"sigmoid", "sigmoid", "sigmoid"});
    mlp.setAccuracy(0.01);

    double learningRate = 0.1;

    mlp.predict(inputs, R_D);

    mlp.train(inputs, targets, learningRate);

    mlp.predict(inputs, R_D);

    return 0;
}

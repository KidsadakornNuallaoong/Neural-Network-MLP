#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// #include "../Library/Perceptron/Perceptron.hpp"
#include "./Library/MLP/MLP.hpp"

int main() {

    // * Test Multi-Layer Perceptron
    vector<int> layersSize = {2, 8, 8, 1};
    MultiLayerPerceptron<double> mlp(layersSize);

    // * XOR Problem
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};

    // vector<vector<double>> inputs = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    // vector<vector<double>> targets = {{3}, {5}, {7}, {9}};

    // mlp.setLayerWeights(0, {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}});
    // mlp.setLayerWeights(1, {{0.7, 0.8, 0.9}});
    // mlp.setLayerBias(0, {0.1, 0.2, 0.3});
    // mlp.setLayerBias(1, {0.4});
    mlp.setActivation({"relu", "relu", "sigmoid"});
    mlp.setAccuracy(0.1);
    mlp.display();

    double learningRate = 0.1;
    int iterations = 0;

    cout << "Initial outputs:" << endl;
    for (const auto& input : inputs) {
        cout << "Input: ";
        for (const auto& i : input) {
            cout << i << " ";
        }
        cout << "Output: " << mlp.feedForward(input)[0] << endl;
    }

    cout << "Training..." << endl;

    mlp.train(inputs, targets, learningRate);

    // mlp.setActivation({"relu", "step"});
    mlp.display();
    for (const auto& input : inputs) {
        cout << "Input: ";
        for (const auto& i : input) {
            cout << i << " ";
        }
        cout << "Output: " << mlp.feedForward(input)[0] << endl;
    }

    // mlp.resetWeightsBias();
    mlp.display();

    cout << "Initial outputs:" << endl;
    for (const auto& input : inputs) {
        cout << "Input: ";
        for (const auto& i : input) {
            cout << i << " ";
        }
        cout << "Output: " << round(mlp.feedForward(input)[0]) << endl;
    }

    // mlp.display();

    return 0;
}

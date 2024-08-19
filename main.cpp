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
    vector<int> layersSize = {2, 3, 1};
    MultiLayerPerceptron<double> mlp(layersSize);

    // * XOR Problem
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};

    // Randomize weights and biases

    mlp.setActivation({"relu", "sigmoid"});
    mlp.setAccuracy(0.01);
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

    cout << endl;
    cout << "Training finished!" << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Accuracy: " << mlp.calculateAccuracy(inputs, targets) * 100 << "%" << endl;
    cout << "Loss: " << mlp.calculateLoss(inputs, targets) << endl;
    cout << "All outputs correct: " << mlp.allOutputsCorrect(inputs, targets) << endl;
    cout << endl;

    mlp.display();
    for (const auto& input : inputs) {
        cout << "Input: ";
        for (const auto& i : input) {
            cout << i << " ";
        }
        cout << "Output: " << mlp.feedForward(input)[0] << endl;
    }


    // mlp.display();

    return 0;
}

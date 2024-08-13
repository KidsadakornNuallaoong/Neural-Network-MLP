#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "./Library/MLP/MLP.hpp"

using namespace std;

int main() {
    // * Test Multi-Layer Perceptron
    vector<int> layersSize = {2, 3, 1};
    MultiLayerPerceptron<double> mlp(layersSize);
    
    // * XOR Problem
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};
    mlp.setLayerWeights(0, {{31.3322, -2.01753}, {0.5, 0.5}, {-276.281, 7.45239}});
    mlp.setLayerBias(0, {-30.1578, 0.5, -2.38014});
    mlp.setLayerWeights(1, {{2.29117, 0.5, 5.90903}});
    mlp.setLayerBias(1, {-2.60602});
    mlp.display();
    double learningRate = 0.1;
    int iterations = 0;

    cout << "Training..." << endl;
    while (!mlp.allOutputsCorrect(inputs, targets)) {
        int index = rand() % inputs.size();
        mlp.train(inputs[index], targets[index], learningRate);
        iterations++;
    }


    cout << "Training finished!" << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Accuracy: " << mlp.calculateAccuracy(inputs, targets) * 100 << "%" << endl;
    cout << "Loss: " << mlp.calculateLoss(inputs, targets) << endl;
    cout << "All outputs correct: " << mlp.allOutputsCorrect(inputs, targets) << endl;
    
    cout << endl;

    cout << "Predictions:" << endl;
    for(const auto& input : inputs) {
        cout << "Input: ";
        for(const auto& i : input) {
            cout << i << " ";
        }
        cout << "Output: " << round(mlp.feedForward(input)[0]) << endl;
    }

    mlp.display();

    cout << endl;

    // * Clone Multi-Layer Perceptron
    cout << "Cloning Multi-Layer Perceptron..." << endl;
    MultiLayerPerceptron<double> mlp2 = mlp.clone();
    for(const auto& input : inputs) {
        cout << "Input: ";
        for(const auto& i : input) {
            cout << i << " ";
        }
        cout << "Output: " << round(mlp2.feedForward(input)[0]) << endl;
    }

    mlp2.display();

    return 0;
}

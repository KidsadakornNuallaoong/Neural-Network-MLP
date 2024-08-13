#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
// #include "../Library/Perceptron/Perceptron.hpp"
#include "./Library/MLP/MLP.hpp"

using namespace std;

int main() {

    // * Test Multi-Layer Perceptron
    vector<int> layersSize = {2, 3, 1};
    MultiLayerPerceptron<double> mlp(layersSize);
    
    // * XOR Problem
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};
    mlp.setLayerWeights(0, {{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}});
    mlp.setLayerBias(0, {0.5, 0.5, 0.5});
    mlp.setLayerWeights(1, {{0.5, 0.5, 0.5}});
    mlp.setLayerBias(1, {0.5});
    mlp.display();
    double learningRate = 0.1;
    int iterations = 0;

    cout << "Training..." << endl;
    while (!mlp.allOutputsCorrect(inputs, targets)) {
        int index = rand() % inputs.size();
        mlp.train(inputs[index], targets[index], learningRate);
        iterations++;
    }
    
    // * alert when training finished

    cout << "Training finished!" << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Accuracy: " << mlp.calculateAccuracy(inputs, targets) * 100 << "%" << endl;
    cout << "Loss: " << mlp.calculateLoss(inputs, targets) << endl;
    cout << "All outputs correct: " << mlp.allOutputsCorrect(inputs, targets) << endl;
    
    for(const auto& input : inputs) {
        cout << "Input: ";
        for(const auto& i : input) {
            cout << i << " ";
        }
        cout << "Output: " << round(mlp.feedForward(input)[0]) << endl;
    }

    mlp.display();

    return 0;
}

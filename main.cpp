#include <iostream>
#include <vector>

using namespace std;

// #include "../Library/Perceptron/Perceptron.hpp"
#include "./Library/MLP/MLP.hpp"

int main() {

    // * Test Multi-Layer Perceptron
    vector<int> layersSize = {2, 8, 8, 3};
    MultiLayerPerceptron<double> mlp(layersSize);

    // * XOR Problem
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0, 0, 0}, {0, 1, 1}, {0, 1, 1}, {1, 1, 0}};

    // vector<vector<double>> inputs = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    // vector<vector<double>> targets = {{0.3}, {0.5}, {0.7}, {0.9}};

    // mlp.setLayerWeights(0, {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}});
    // mlp.setLayerWeights(1, {{0.7, 0.8, 0.9}});
    // mlp.setLayerBias(0, {0.1, 0.2, 0.3});
    // mlp.setLayerBias(1, {0.4});
    mlp.setActivation({"sigmoid", "sigmoid", "sigmoid"});
    mlp.setAccuracy(0.1);
    mlp.display();

    double learningRate = 0.1;

    cout << "Initial outputs:" << endl;
    mlp.predict(inputs, DISPLAY);

    cout << "Training..." << endl;

    mlp.train(inputs, targets, learningRate);

    // mlp.resetWeightsBias();
    mlp.display();

    cout << "Initial outputs:" << endl;
    mlp.predict(inputs, R_D);

    // // mlp.display();
    // mlp.export_to_json("mlp.json");

    // vector<int> layersSize = {2, 8, 8, 1};
    // MultiLayerPerceptron<double> mlp2(layersSize);
    // mlp2.setActivation({"relu", "relu", "sigmoid"});
    // mlp2.setAccuracy(0.1);
    // // mlp2.display();
    // mlp2.import_from_json("mlp.json");

    // cout << "Initial outputs:" << endl;
    // vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    // for (const auto& input : inputs) {
    //     cout << "Input: ";
    //     for (const auto& i : input) {
    //         cout << i << " ";
    //     }
    //     cout << "Output: " << mlp2.feedForward(input)[0] << endl;
    // }

    return 0;
}

// ****************************************************
// * Code by Kidsadakorn Nuallaoong
// * Multi-Layer Perceptron
// * Artificial Neuron Network - Multi Layer
// ****************************************************

#if !defined(MLP_HPP)
#define MLP_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>

#include "../Perceptron/Perceptron.hpp"

using namespace std;

template <typename T>
class MultiLayerPerceptron : private Perceptron<T>
{
    private:
        // * Perceptron layers in vector
        vector<vector<Perceptron<T>>> layers;

        string activationType = "sigmoid";

    public:
        // * Constructor
        MultiLayerPerceptron();

        // * Constructor with layers size
        MultiLayerPerceptron(const vector<int>& layersSize);
        
        // * Destructor
        ~MultiLayerPerceptron();

        // * Initialize layers
        void initLayer(const vector<int>& layersSize);

        // TODO : feedForward, train, display
        vector<T> feedForward(const vector<T>& inputs);
        void train(const vector<T>& inputs, const vector<T>& targets, const T learningRate);

        void typeDeActivation(string type);
        T activationDerivative(T x);

        // TODO : Calculate accuracy, loss, all_outputs_correct
        T calculateAccuracy(const vector<vector<T>>& inputs, const vector<vector<T>>& targets);
        T calculateLoss(const vector<vector<T>>& inputs, const vector<vector<T>>& targets);
        bool allOutputsCorrect(const vector<vector<T>>& inputs, const vector<vector<T>>& targets);
        
        // TODO : setLayerWeights, setLayerBias
        void setLayerWeights(int layerIndex, const vector<vector<T>>& weights);
        void setLayerBias(int layerIndex, const vector<T>& bias);

        // * Display layers
        void display();

        // * Clone the MLP
        MultiLayerPerceptron<T> clone() const;
};

#endif // MLP_HPP
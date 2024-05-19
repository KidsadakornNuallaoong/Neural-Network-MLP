#if !defined(NEURAL_H)
#define NEURAL_H

#include <iostream>
#include "../Perceptron/Perceptron.h"

using namespace std;

typedef struct HiddenLayer {
    int hiddenCount;
    int hiddenNeurons;
} HiddenLayer;

template <typename T>

class Neural : public Perceptron<T>{
    private:
        vector<T> inputs;
        vector<vector<T>> weights;
        vector<vector<Perceptron<T>>> perceptrons = {{}};
        vector<vector<T>> bias = {{0}};
        vector<vector<T>> biasWeight = {{0}};

        HiddenLayer hiddenLayer = {0, 0};
        string activationFunction = "step";

    public:
        Neural();
        ~Neural();

        void setInputs(vector<T> inputs, bool verbose = false);
        void setWeights(vector<vector<T>> weights, bool verbose = false);
        void setBiasWeight(vector<vector<T>> biasWeights, bool verbose = false);
        void setHiddenLayer();
        void MultiLayerPerceptron(int layer = 1);

        void display(string option = "neuron");
        void displayNeurons();
        void displayPerceptrons();
};


#endif // NEURAL_H
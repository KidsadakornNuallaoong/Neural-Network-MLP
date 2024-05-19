#if !defined(NEURAL_H)
#define NEURAL_H

#include <iostream>
#include "../Perceptron/Perceptron.h"

using namespace std;
template <typename T>

class Neural {
    private:
        vector<T> inputs;
        vector<vector<T>> weights;
        vector<Perceptron<T>> perceptrons;
        vector<T> biass;
        vector<T> biasWeights;
        vector<T> targets;

        T bias = 1;
        T biasWeight = 1;
        T learningRate = 0.1;
        T accuracy = 0.0001;
        string activationFunction = "step";

    public:
        Neural();
        ~Neural();

        void setInputs(vector<T> inputs, bool verbose = false);
        void setWeights(vector<vector<T>> weights, bool verbose = false);
        void setHiddenLayer();

        void display();
};


#endif // NEURAL_H
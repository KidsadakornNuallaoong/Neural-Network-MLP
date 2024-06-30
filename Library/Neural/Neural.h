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
};


#endif // NEURAL_H
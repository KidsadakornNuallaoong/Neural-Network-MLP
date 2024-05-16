#if !defined(PERCEPTRON_H)
#define PERCEPTRON_H

#include <vector>
#include <iostream>

using namespace std;

template <typename T>

class Perceptron
{
    private:
        vector<T> inputs;
        vector<T> weights;
        T bias;
        T biasWeight;
        T output = 0;
        T target;
        T learningRate;
        T accuracy = 0.001;
        T error;
        string activationFunction = "step";

        int count = 0;

        // * formula
        T stepFunction(T x);
        T signFunction(T x);
        T sigmoidFunction(T x);
        T tanhFunction(T x);
        T reluFunction(T x);
        T leakyReluFunction(T x);
        T eluFunction(T x);
        T softplusFunction(T x);
        T softsignFunction(T x);
        T gaussianFunction(T x);

        T sineFunction(T x);
        T cosineFunction(T x);  
        T arctanFunction(T x);

        T linearFunction(T x);

        T activation(T x);
    public:
        Perceptron();
        Perceptron(vector<T> inputs, vector<T> weights = {0}, T bias = 1, T biasWeight = 1, T learningRate = 0.1, T target = 0, T error = 0, T output = 0);
        ~Perceptron();

        // * set up
        void setInputs(vector<T> inputs);
        void setWeights(vector<T> weights);
        void setBias(T bias);
        void setBiasWeight(T biasWeight);
        void setLearningRate(T learningRate);
        void setTarget(T target);
        void setError(T error);
        void setAccuracy(T accuracy);

        // * get
        vector<T> getInputs();
        vector<T> getWeights();
        T getBias();
        T getOutput();
        T getTarget();
        T getLearningRate();
        T getError();

        // * functions
        void setActivationFunction(string activationFunction);
        T feedForward();
        void train(bool verbose = false);

        // * monitor
        void display();
};

#endif // PERCEPTRON_H

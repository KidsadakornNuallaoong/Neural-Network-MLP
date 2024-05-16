#if !defined(PERCEPTRON_H)
#define PERCEPTRON_H

#include <vector>
#include <iostream>

using namespace std;
enum ActivationFunction { STEP, SIGN, SIGMOID, TANH, RELU, LEAKY_RELU, ELU, SOFTPLUS, SOFTSIGN, GAUSSIAN, SINE, COSINE, ARCTAN, LINEAR };

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
        T error;

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

        T activation(T x, ActivationFunction f = STEP);
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

        // * get
        vector<T> getInputs();
        vector<T> getWeights();
        T getBias();
        T getOutput();
        T getTarget();
        T getLearningRate();
        T getError();

        // * functions
        void feedForward(bool display = false);
        void backPropagation(bool display = false);
        void train(bool display = false);

        // * monitor
        void display();
};

#endif // PERCEPTRON_H

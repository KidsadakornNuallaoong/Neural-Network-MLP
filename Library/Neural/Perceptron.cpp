#include "perceptron.h"
#include <cmath>
#include <iomanip>

using namespace std;
using std::fixed;
using std::vector;

template <typename T>
T Perceptron<T>::stepFunction(T x)
{
    return T(x >= 0 ? 1 : 0);
}

template <typename T>
T Perceptron<T>::signFunction(T x)
{
    return T(x >= 0 ? 1 : x < 0 ? -1 : 0);
}

template <typename T>
T Perceptron<T>::sigmoidFunction(T x)
{
    return T(1 / (1 + exp(-x)));
}

template <typename T>
T Perceptron<T>::tanhFunction(T x)
{
    return T(tanh(x));
}

template <typename T>
T Perceptron<T>::reluFunction(T x)
{
    return T(x >= 0 ? x : 0);
}

template <typename T>
T Perceptron<T>::leakyReluFunction(T x)
{
    return T(x >= 0 ? x : 0.01 * x);
}

template <typename T>
T Perceptron<T>::eluFunction(T x)
{
    return T(x >= 0 ? x : 0.01 * (exp(x) - 1));
}

template <typename T>
T Perceptron<T>::softplusFunction(T x)
{
    return T(log(1 + exp(x)));
}

template <typename T>
T Perceptron<T>::softsignFunction(T x)
{
    return T(x / (1 + abs(x)));
}

template <typename T>
T Perceptron<T>::linearFunction(T x)
{
    return T(x);
}

template <typename T>
T Perceptron<T>::gaussianFunction(T x)
{
    return T(exp(-x * x));
}

template <typename T>
T Perceptron<T>::sineFunction(T x)
{
    return T(sin(x));
}

template <typename T>
T Perceptron<T>::cosineFunction(T x)
{
    return T(cos(x));
}

template <typename T>
T Perceptron<T>::arctanFunction(T x)
{
    return T(atan(x));
}

template <typename T>
inline Perceptron<T>::Perceptron()
{
    cout << "Perceptron created" << endl;
    // cout << std::fixed << std::setprecision(5);
}

template <typename T>
Perceptron<T>::Perceptron(vector<T> inputs, vector<T> weights, T bias, T biasWeight, T learningRate, T target, T error, T output)
{
    // cout << std::fixed << std::setprecision(5);
    this->inputs = inputs;
    this->weights = weights;
    this->bias = bias;
    this->biasWeight = biasWeight;
    this->learningRate = learningRate;
    this->target = target;
    this->error = error;
    cout << "Perceptron created" << endl;
}

template <typename T>
inline Perceptron<T>::~Perceptron()
{
    cout << "Perceptron destroyed" << endl;
}

template <typename T>
void Perceptron<T>::setInputs(vector<T> inputs)
{
    this->inputs = inputs;
}

template <typename T>
void Perceptron<T>::setWeights(vector<T> weights)
{
    this->weights = weights;
}

template <typename T>
void Perceptron<T>::setBias(T bias)
{
    this->bias = bias;
}

template <typename T>
void Perceptron<T>::setBiasWeight(T biasWeight)
{
    this->biasWeight = biasWeight;
}

template <typename T>
void Perceptron<T>::setLearningRate(T learningRate)
{
    this->learningRate = learningRate;
}

template <typename T>
void Perceptron<T>::setTarget(T target)
{
    this->target = target;
}

template <typename T>
void Perceptron<T>::setError(T error)
{
    this->error = error;
}

template <typename T>
void Perceptron<T>::setAccuracy(T accuracy)
{
    this->accuracy = accuracy;
}

template <typename T>
vector<T> Perceptron<T>::getInputs()
{
    return vector<T>(inputs);
}

template <typename T>
vector<T> Perceptron<T>::getWeights()
{
    return vector<T>(weights);
}

template <typename T>
T Perceptron<T>::getBias()
{
    return T(bias);
}

template <typename T>
T Perceptron<T>::getOutput()
{
    return T(output);
}

template <typename T>
T Perceptron<T>::getTarget()
{
    return T(target);
}

template <typename T>
T Perceptron<T>::getLearningRate()
{
    return T(learningRate);
}

template <typename T>
T Perceptron<T>::getError()
{
    return T(error);
}

template <typename T>
T Perceptron<T>::getAccuracy()
{
    return T(accuracy);
}

template <typename T>
T Perceptron<T>::activation(T x)
{
    string ActF = this->activationFunction;
    for(char &c : ActF) c = tolower(c);
    if (ActF == "step")
    {
        return stepFunction(x);
    }
    else if (ActF == "sign")
    {
        return signFunction(x);
    }
    else if (ActF == "sigmoid")
    {
        return sigmoidFunction(x);
    }
    else if (ActF == "tanh")
    {
        return tanhFunction(x);
    }
    else if (ActF == "relu")
    {
        return reluFunction(x);
    }
    else if (ActF == "leakyRelu")
    {
        return leakyReluFunction(x);
    }
    else if (ActF == "elu")
    {
        return eluFunction(x);
    }
    else if (ActF == "softplus")
    {
        return softplusFunction(x);
    }
    else if (ActF == "softsign")
    {
        return softsignFunction(x);
    }
    else if (ActF == "gaussian")
    {
        return gaussianFunction(x);
    }
    else if (ActF == "sine")
    {
        return sineFunction(x);
    }
    else if (ActF == "cosine")
    {
        return cosineFunction(x);
    }
    else if (ActF == "arctan")
    {
        return arctanFunction(x);
    }
    else
    {
        return linearFunction(x);
    }
}

template <typename T>
void Perceptron<T>::setActivationFunction(string activationFunction)
{
    this->activationFunction = activationFunction;
}

template <typename T>
T Perceptron<T>::feedForward()
{
    this->output = 0;
    for (int i = 0; i < this->inputs.size(); i++)
    {
        this->output += this->inputs[i] * this->weights[i];
    }
    this->output += this->bias * this->biasWeight;
    this->output = activation(this->output);
    this->error = (this->target - this->output);
    return this->output;
}

template <typename T>
void Perceptron<T>::train(bool verbose)
{
    do {
        this->error = (this->target - this->output);
        T alpha = this->learningRate * this->error;
        for (int i = 0; i < this->inputs.size(); i++)
        {
            // * w = w + Δw
            // * Δw = η * error * input
            this->weights[i] = this->weights[i] +  alpha * this->inputs[i];
        }
        this->biasWeight = this->biasWeight + alpha * this->bias;
        count++;
        this->feedForward();
    } while (this->error >= (this->accuracy * 1) || this->error <= (this->accuracy * -1));
}

// template <typename T>
// void Perceptron<T>::train(bool verbose)
// {
//     // * set time start
//     float start = clock();
//     // * setprecision(2) for float
//     do
//     {
//         this->feedForward();
//         this->error = (this->target - this->output);
//         for(int i = 0; i < this->inputs.size(); i++)
//         {
//             // * w = w + Δw
//             // * Δw = η * error * input
//             this->weights[i] += this->error * this->learningRate * this->inputs[i];
//         }
//         this->biasWeight += this->error * this->learningRate * this->bias;
//         count++;
//     } while (this->error >= (this->accuracy * 1) || this->error <= (this->accuracy * -1));
//     // * set time end
//     float end = clock();

//     if (verbose)
//     {
//         cout << "Time: " << (end - start) / CLOCKS_PER_SEC << "s" << endl;
//         cout << ">> Training End <<" << endl << endl;
//     }
// }

template <typename T>
void Perceptron<T>::display()
{
    cout << ">> Perceptron Display <<" << endl;
    cout << "Inputs: ";
    for (int i = 0; i < this->inputs.size(); i++)
    {
        cout << this->inputs[i] << " ";
    }
    cout << endl;

    cout << "Weights: ";
    for (int i = 0; i < this->weights.size(); i++)
    {
        cout << this->weights[i] << " ";
    }
    cout << endl;
    cout << "Epoch: " << this->count << endl;
    cout << "Bias: " << this->bias << " ";
    cout << "Bias Weight: " << this->biasWeight << endl;
    cout << "Learning Rate: " << this->learningRate << " ";
    cout << "Error Rate: " << (this->target - this->output) << endl;
    cout << "Activation Function: " << this->activationFunction << " ";
    cout << "Accuracy: " << this->accuracy << endl;
    cout << "Target: " << this->target << " Model Answer: " << this->output << endl << endl;
}

// Explicitly instantiate the template for the types you need
template class Perceptron<double>;
template class Perceptron<float>;
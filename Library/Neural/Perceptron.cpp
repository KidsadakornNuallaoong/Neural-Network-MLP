#include "perceptron.h"
#include <cmath>

using namespace std;

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
}

template <typename T>
Perceptron<T>::Perceptron(vector<T> inputs, vector<T> weights, T bias, T biasWeight, T learningRate, T target, T error, T output)
{
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
void Perceptron<T>::feedForward(bool display)
{
    T sum = 0;
    for (int i = 0; i < this->inputs.size(); i++)
    {
        sum += this->inputs[i] * this->weights[i];
        if(display){
            printf("Root [%d] = %f ", i, this->inputs[i] * this->weights[i]);
            printf("Sum = %f\n\n", sum);
        }
    }
    sum += this->bias * this->biasWeight;
    if(display){
        cout << "Bias: " << this->bias << " Bias Weight: " << this->biasWeight << endl;
        cout << "Sum: " << sum << endl;
    }
    this->output = activation(sum);
    if(display){
        cout << "Output: " << this->output << endl;
    }
}

template <typename T>
void Perceptron<T>::backPropagation(bool display)
{
    error = target - output;
    for(int i = 0; i < weights.size(); i++){
        this->weights[i] += learningRate * error * inputs[i];
    }
    this->bias += learningRate * error * biasWeight;
}

template <typename T>
void Perceptron<T>::train(bool display)
{   
    string ans;
    do
    {
        count = count + 1;
        feedForward();
        backPropagation();
        if(display){
            // * use this to display the training process
            Perceptron<T>::display();
        }
        cout << "Target: " << target << " Output: " << output << endl;
        cout << "it is correct? (Y/N): ";
        cin >> ans;
    } while (ans != "Y" && ans != "y");
}

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
    cout << "Epoch: " << count << " times\n";
    cout << "Bias: " << this->bias << endl;
    cout << "Learning Rate: " << this->learningRate << " ";
    cout << "Error: " << this->error << endl;
    cout << "Target: " << this->target << " Output: " << this->output << endl << endl;
}

// Explicitly instantiate the template for the types you need
template class Perceptron<double>;
template class Perceptron<float>;
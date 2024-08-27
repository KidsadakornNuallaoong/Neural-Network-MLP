#include "Perceptron.hpp"
#include <cmath>
#include <string>
#include <chrono>

using namespace std;
using std::vector;

template <typename T>
Perceptron<T>::Perceptron()
{
    // * Constructor
    this->bias = 0;
    this->output = 0;
    this->activationType = "sigmoid";
}

template <typename T>
Perceptron<T>::Perceptron(int inputSize)
{
    init(inputSize);
}

template <typename T>
Perceptron<T>::~Perceptron()
{
    // * Destructor
    // * Free memory
    weights.clear();
    weights.shrink_to_fit();

    // * Reset bias
    bias = 0;
    output = 0;
    activationType = "sigmoid";
}

template <typename T>
void Perceptron<T>::init(int inputSize)
{
    weights.resize(inputSize);
    for (int i = 0; i < inputSize; ++i) {
        weights[i] = ((T)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
    }
    bias = ((T)rand() / RAND_MAX) * 2 - 1;
}

template <typename T>
void Perceptron<T>::setWeights(const vector<T>& weights) {
    this->weights = weights;
}

template <typename T>
void Perceptron<T>::setWeights(const int index, const T weights)
{
    this->weights[index] = weights;
}

template <typename T>
void Perceptron<T>::setBias(T bias)
{
    this->bias = T(bias);
}

template <typename T>
void Perceptron<T>::resetWeightsBias()
{
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] = ((T)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
    }
    bias = ((T)rand() / RAND_MAX) * 2 - 1;
}

template <typename T>
vector<T> Perceptron<T>::_weights()
{
    return vector<T>(this->weights);
}

template <typename T>
T Perceptron<T>::_bias()
{
    return T(this->bias);
}

template <typename T>
void Perceptron<T>::typeActivation(string type)
{
    // * change type to lower case
    for (int i = 0; i < type.length(); i++)
    {
        type[i] = tolower(type[i]);
    }

    if (!(type == "linear" || 
          type == "sigmoid" ||
          type == "tanh" ||
          type == "relu" ||
          type == "leakyrelu" ||
          type == "softmax" ||
          type == "step"))
    {
        cerr << "\033[1;31mActivation Type Not Found\033[0m" << endl;
        return;
    }

    this->activationType = type;
}

template <typename T>
T Perceptron<T>::activation(T x) {
    if (activationType == "linear") {
        return x;
    } else if (activationType == "sigmoid") {
        return 1 / (1 + exp(-x));
    } else if (activationType == "tanh") {
        return tanh(x);
    } else if (activationType == "relu") {
        return (x > 0) ? x : 0;
    } else if (activationType == "leakyrelu") {
        return (x > 0) ? x : 0.01 * x;
    } else if (activationType == "softmax") {
        // Softmax is not applicable here for scalar values.
        std::cerr << "\033[1;31mSoftmax must be computed over a vector, not a scalar.\033[0m" << std::endl;
        throw std::invalid_argument("Softmax must be computed over a vector.");
    } else if (activationType == "step") {
        return (x > 0) ? 1 : 0;
    } else {
        std::cerr << "\033[1;31mActivation Type Not Found\033[0m" << std::endl;
        throw std::invalid_argument("Activation Type Not Found");
    }
}

template <typename T>
T Perceptron<T>::feedForward(const vector<T>& inputs)
{
    T total = bias;
    for (int i = 0; i < weights.size(); ++i) {
        total += weights[i] * inputs[i];
    }

    output = activation(total);
    return T(output);
}

template <typename T>
T Perceptron<T>::feedForward(const vector<T>& inputs, T bias)
{
    T total = bias;
    for (int i = 0; i < weights.size(); ++i) {
        total += weights[i] * inputs[i];
    }

    output = activation(total);
    return T(output);
}

template <typename T>
void Perceptron<T>::train(const vector<T>& inputs, T target, const T learningRate) {
    T error = target - feedForward(inputs);
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] += learningRate * error * inputs[i];
    }
    bias += learningRate * error;
}

template <typename T>
vector<T> Perceptron<T>::getWeights()
{
    return vector<T>(this->weights);
}

template <typename T>
T Perceptron<T>::getBias()
{
    return T(this->bias);
}

template <typename T>
Perceptron<T> Perceptron<T>::cpyEnv() const
{
    return Perceptron<T>(*this);
}

template <typename T>
T Perceptron<T>::getOutput()
{
    return T(this->output);
}

template <typename T>
void Perceptron<T>::display()
{
    cout << "\033[1;32m-->> Perceptron <<--\033[0m" << endl << endl;
    cout << "\033[1;33mSize:\033[0m " << weights.size() << endl;
    cout << "\033[1;33mWeights:\033[0m ";
    for (int i = 0; i < weights.size(); ++i) {
        cout << weights[i] << " ";
    }
    cout << endl;
    cout << "\033[1;33mBias:\033[0m " << bias << endl;
    cout << "\033[1;33mActivation Type:\033[0m " << activationType << endl;
    cout << "\033[1;33mOutput:\033[0m " << output << endl;
}

// Explicitly instantiate the template for the types you need
template class Perceptron<long>;
template class Perceptron<double>;
template class Perceptron<float>;
template class Perceptron<int>;
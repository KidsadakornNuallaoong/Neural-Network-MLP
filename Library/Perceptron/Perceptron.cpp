#include "Perceptron.h"
#include <cmath>
#include <string>
#include <chrono>

using namespace std;
using std::vector;

template <typename T>
Perceptron<T>::Perceptron(){}

template <typename T>
Perceptron<T>::~Perceptron(){}

template <typename T>
void Perceptron<T>::setInputs(vector<T> inputs) {
    this->inputs = inputs;
}

template <typename T>
void Perceptron<T>::setWeights(vector<T> weights) {
    this->weights = weights;
}

template <typename T>
void Perceptron<T>::setBias(T biasW) {
    this->biasW = biasW;
}
template <typename T>
void Perceptron<T>::setLearningRate(T learningRate) {
    this->learningRate = learningRate;
}
template <typename T>
void Perceptron<T>::setTarget(T target) {
    this->target = target;
}

template <typename T>
vector<T> Perceptron<T>::getInputs()
{
    return vector<T>(this->inputs);
}

template <typename T>
vector<T> Perceptron<T>::getWeights()
{
    return vector<T>(this->weights);
}

template <typename T>
T Perceptron<T>::getBias()
{
    return T(this->biasW);
}

template <typename T>
T Perceptron<T>::getLearningRate()
{
    return T(this->learningRate);
}

template <typename T>
T Perceptron<T>::getTarget()
{
    return T(this->target);
}

template <typename T>
T Perceptron<T>::getOutput()
{
    return T(this->output);
}

template <typename T>
void Perceptron<T>::copyEnv(Perceptron<T> *p) {
    this->weights = p->getWeights();
    this->biasW = p->getBias();
    this->learningRate = p->getLearningRate();
    this->target = p->getTarget();
}

template <typename T>
T Perceptron<T>::Err()
{
    return T(this->error);
}

template <typename T>
T Perceptron<T>::MSE()
{
    return T(pow(Err(),2)/2);
}

template <typename T>
T Perceptron<T>::MAE()
{
    return T(abs(Err()));
}

template <typename T>
T Perceptron<T>::activation(T x)
{
    return T(x);
}

template <typename T>
void Perceptron<T>::setActivation(string type)
{
    // * change type to lower case
    for (int i = 0; i < type.length(); i++)
    {
        type[i] = tolower(type[i]);
    }
    this->activationType = type;
}

template <typename T>
T Perceptron<T>::feedForward()
{
    // * calculate the output
    // * output = ∑(inputs * weights) + bias
    this->output = 0;
    for (int i = 0; i < this->inputs.size(); i++)
    {
        // * ∑(inputs * weights)
        this->output += this->inputs[i] * this->weights[i];
    }
    // * + bias
    this->output += this->biasW;

    // * e = out - target
    this->error = this->output - this->target;
    return activation(this->output);
}

template <typename T>
T Perceptron<T>::backpropagate()
{
    // * calculate updated weights
    // * w' = w - η * ∂E/∂w
    // * b' = b - η * ∂E/∂b
    // * ∂E/∂w = 2 * (target - output) * input
    // * ∂E/∂b = 2 * (target - output)
    // * E = 1/2 * (target - output)^2
    // * η = learning rate
    for (int i = 0; i < this->weights.size(); i++)
    {
        this->weights[i] = this->weights[i] - this->learningRate * 2 * this->error * this->inputs[i];
    }
    this->biasW = this->biasW - this->learningRate * 2 * this->error;
    this->output = feedForward();
    return this->error;
}

template <typename T>
void Perceptron<T>::train(bool verbose)
{
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++)
    {
        backpropagate();
        if (verbose)
        {
            display();
        }
        if (Err() < 0.0000001 && Err() > -0.0000001)
        {
            break;
        }
    }
    auto end = chrono::high_resolution_clock::now();
    double TrainTime = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    if(verbose){
        cout << "Train Time: " << TrainTime << " ns" << endl;
    }
}

template <typename T>
void Perceptron<T>::train(int epoch, bool verbose)
{
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < epoch; i++)
    {
        backpropagate();
        if (verbose)
        {
            display();
        }
        if (Err() < 0.0000001 && Err() > -0.0000001)
        {
            cout << "Error: " << Err() << endl;
            break;
        }
    }
    auto end = chrono::high_resolution_clock::now();
    double TrainTime = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    if(verbose){
        cout << "Train Time: " << TrainTime << " ns" << endl;
    }
}

template <typename T>
void Perceptron<T>::display()
{
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

    cout << "Bias: " << this->biasW << endl;
    cout << "Act: " << this->activationType << " ";
    cout << "Learning Rate: " << this->learningRate << endl;
    cout << "Target: " << this->target << endl;
    cout << "Output: " << this->output << " ";
    cout << "Error: " << this->error << endl;
    cout << "MSE: " << this->MSE() << " ";
    cout << "MAE: " << this->MAE() << endl;
}

// Explicitly instantiate the template for the types you need
template class Perceptron<double>;
template class Perceptron<float>;
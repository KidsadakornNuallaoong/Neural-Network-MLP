#include "MLP.hpp"

template <typename T>
MultiLayerPerceptron<T>::MultiLayerPerceptron()
{
    // * Set seed for random
    srand((unsigned int)time(NULL));
    // * Constructor
    this->layers = vector<vector<Perceptron<T>>>();
}

template <typename T>
MultiLayerPerceptron<T>::MultiLayerPerceptron(const vector<int> &layersSize)
{
    // * Constructor
    this->layers = vector<vector<Perceptron<T>>>();
    initLayer(layersSize);
}

template <typename T>
MultiLayerPerceptron<T>::~MultiLayerPerceptron()
{
    // * Destructor
    // * Free memory
    for (int i = 0; i < layers.size(); ++i) {
        layers[i].clear();
        layers[i].shrink_to_fit();
    }
    layers.clear();
    layers.shrink_to_fit();
}

template <typename T>
void MultiLayerPerceptron<T>::initLayer(const vector<int>& Size)
{
    // * Set seed for random
    srand((unsigned int)time(NULL));
    // * Initialize layers
    for (int i = 0; i < Size.size() - 1; ++i) {
        layers.push_back(vector<Perceptron<T>>());
        for (int j = 0; j < Size[i + 1]; ++j) {
            layers[i].push_back(Perceptron<T>(Size[i]));
        }
    }
}

template <typename T>
vector<T> MultiLayerPerceptron<T>::feedForward(const vector<T> &inputs)
{
    vector<T> outputs = inputs;
    for (int i = 0; i < layers.size(); ++i) {
        vector<T> newOutputs;
        for (int j = 0; j < layers[i].size(); ++j) {
            newOutputs.push_back(layers[i][j].feedForward(outputs));
        }
        outputs = newOutputs;
    }
    return outputs;
}

template <typename T>
void MultiLayerPerceptron<T>::train(const vector<T> &inputs, const vector<T> &targets, const T learningRate)
{
    vector<vector<T>> layerOutputs;
    vector<T> outputs = inputs;
    layerOutputs.push_back(outputs);

    // * Feed Forward
    for (int i = 0; i < layers.size(); ++i) {
        outputs = feedForward(outputs);
        layerOutputs.push_back(outputs);
    }

    // * Calcualte Errors
    vector<T> errors;
    for (int i = 0; i < targets.size(); ++i) {
        errors.push_back(targets[i] - layerOutputs.back()[i]);
    }

    // * Back Propagation
    for (int i = layers.size() - 1; i >= 0; --i) {
        vector<T> newErrors(layers[i][0].weights.size(), 0);

        #pragma omp parallel for
        for (int j = 0; j < layers[i].size(); ++j) {
            T error = errors[j] * activationDerivative(layerOutputs[i + 1][j]);
            for (int k = 0; k < layers[i][j].weights.size(); ++k) {
                layers[i][j].weights[k] += error * layerOutputs[i][k] * learningRate;
                #pragma omp atomic
                newErrors[k] += layers[i][j].weights[k] * error;
            }
            layers[i][j].bias += error * learningRate;
        }

        errors = newErrors;
    }
}

template <typename T>
void MultiLayerPerceptron<T>::typeDeActivation(string type)
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
T MultiLayerPerceptron<T>::activationDerivative(T x)
{
    if (activationType == "linear")
    {
        // * f(x) = x
        return 1;
    }
    else if (activationType == "sigmoid")
    {
        // * f(x) = 1 / (1 + e^(-x))
        return x * (1 - x);
    }
    else if (activationType == "tanh")
    {
        // * f(x) = tanh(x)
        return 1 - x * x;
    }
    else if (activationType == "relu")
    {
        // * f(x) = max(0, x)
        return x > 0 ? 1 : 0;
    }
    else if (activationType == "leakyrelu")
    {
        // * f(x) = max(0.01x, x)
        return x > 0 ? 1 : 0.01;
    }
    else if (activationType == "softmax")
    {
        // * f(x) = e^x / sum(e^x)
        return x * (1 - x);
    }
    else if (activationType == "step")
    {
        // * f(x) = 1 if x > 0 else 0
        return x > 0 ? 1 : 0;
    }else{
        cerr << "\033[1;31mActivation Type Not Found\033[0m" << endl;
        return 0;
    }
}

template <typename T>
T MultiLayerPerceptron<T>::calculateAccuracy(const vector<vector<T>> &inputs, const vector<vector<T>> &targets)
{
    int correct = 0;

    for (int i = 0; i < inputs.size(); ++i) {
        vector<T> output = feedForward(inputs[i]);
        for (int j = 0; j < output.size(); ++j) {
            if (round(output[j]) == targets[i][j]) {
                correct++;
            }
        }
    }

    // * formula : correct / (input(size) * target(size))
    return (T)correct / (inputs.size() * targets[0].size());
}

template <typename T>
T MultiLayerPerceptron<T>::calculateLoss(const vector<vector<T>> &inputs, const vector<vector<T>> &targets)
{
    double totalLoss = 0.0;

    for (int i = 0; i < inputs.size(); ++i) {
        vector<T> output = feedForward(inputs[i]);
        for (int j = 0; j < output.size(); ++j) {
            T error = targets[i][j] - output[j];
            totalLoss += error * error;
        }
    }

    // * formula : totalLoss / (input(size) * target(size))
    return T(totalLoss / (inputs.size() * targets[0].size())); 
}

template <typename T>
bool MultiLayerPerceptron<T>::allOutputsCorrect(const vector<vector<T>> &inputs, const vector<vector<T>> &targets)
{
    for (int i = 0; i < inputs.size(); ++i) {
        vector<T> output = feedForward(inputs[i]);
        for (int j = 0; j < output.size(); ++j) {
            if (round(output[j]) != targets[i][j]) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
void MultiLayerPerceptron<T>::setLayerWeights(int layerIndex, const vector<vector<T>> &weights)
{
    for (int i = 0; i < layers[layerIndex].size(); ++i) {
        layers[layerIndex][i].setWeights(weights[i]);
    }
}

template <typename T>
void MultiLayerPerceptron<T>::setLayerBias(int layerIndex, const vector<T> &bias)
{
    for (int i = 0; i < layers[layerIndex].size(); ++i) {
        layers[layerIndex][i].setBias(bias[i]);
    }
}

template <typename T>
void MultiLayerPerceptron<T>::display()
{
    // * Display layers
    for (int i = 0; i < layers.size(); ++i) {
        cout << "Layer " << i << endl;
        cout << "Node ";
        for (int j = 0; j < layers[i].size(); ++j) {
            cout << j << " ";
            // * display perceptron
            layers[i][j].display();
        }
        cout << endl;
    }

    cout << "Activation Type: " << activationType << endl;
}

template <typename T>
MultiLayerPerceptron<T> MultiLayerPerceptron<T>::clone() const
{
    return MultiLayerPerceptron<T>(*this);
}

// Explicitly instantiate the template for the types you need
template class MultiLayerPerceptron<long>;
template class MultiLayerPerceptron<double>;
template class MultiLayerPerceptron<float>;
template class MultiLayerPerceptron<int>;
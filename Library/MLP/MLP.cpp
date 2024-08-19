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
            layers[i][j].typeActivation(activationTypes[i]);
            newOutputs.push_back(layers[i][j].feedForward(outputs));
        }
        outputs = newOutputs;
    }
    return outputs;
}

// template <typename T>
// void MultiLayerPerceptron<T>::backPropagation(const vector<vector<T>> &inputs, const vector<vector<T>> &targets, const T learningRate)
// {
//     for (int i = 0; i < inputs.size(); ++i) {
//         vector<vector<T>> outputs;
//         vector<vector<T>> errors(layers.size());
//         vector<vector<T>> biases(layers.size());

//         // * feed forward
//         outputs.push_back(inputs[i]);
//         for (int j = 0; j < layers.size(); ++j) {
//             vector<T> newOutputs;
//             for (int k = 0; k < layers[j].size(); ++k) {
//                 layers[j][k].typeActivation(activationTypes[j]);
//                 newOutputs.push_back(layers[j][k].feedForward(outputs.back()));
//             }
//             outputs.push_back(newOutputs);
//         }

//         // * calculate error (start from output layer and move backwards)
//         for (int j = layers.size() - 1; j >= 0; --j) {
//             vector<T> newErrors;
//             if (j == layers.size() - 1) {  // output layer
//                 for (int k = 0; k < layers[j].size(); ++k) {
//                     newErrors.push_back(outputLayerError(outputs[j + 1][k], targets[i][k]));
//                 }
//             } else {  // hidden layers
//                 for (int k = 0; k < layers[j].size(); ++k) {
//                     T error = 0.0;
//                     for (int l = 0; l < layers[j + 1].size(); ++l) {
//                         // * use output error for hidden layer
//                         error += hiddenLayerError(outputs[j + 1][l], errors[j + 1][l], layers[j + 1][l].getWeights()[k]);
//                     }
//                     newErrors.push_back(error);
//                 }
//             }
//             errors[j] = newErrors;
//         }

//         // // * display errors
//         // for (int j = 0; j < errors.size(); ++j) {
//         //     cout << "Layer " << j << " errors: ";
//         //     for (int k = 0; k < errors[j].size(); ++k) {
//         //         cout << errors[j][k] << " ";
//         //     }
//         //     cout << endl;
//         // }

//         // * update weights and bias
//         for (int j = layers.size() - 1; j >= 0; --j) {
//             for (int k = 0; k < layers[j].size(); ++k) {
//                 // cout << "Layer " << j << " Node " << k << endl;
//                 for (int l = 0; l < layers[j][k].getWeights().size(); ++l) {
//                     layers[j][k].setWeights(l, updateWeights(layers[j][k].getWeights()[l], learningRate, errors[j][k], outputs[j][l]));
//                     layers[j][k].setBias(updateBias(layers[j][k].getBias(), learningRate, errors[j][k]));
//                 }
//             }
//         }
//     }
// }

template <typename T>
void MultiLayerPerceptron<T>::backPropagation(const vector<vector<T>> &inputs, const vector<vector<T>> &targets, const T learningRate)
{
    for (int i = 0; i < inputs.size(); ++i) {
        vector<vector<T>> outputs;
        // * feed forward
        outputs.push_back(inputs[i]);
        for (int j = 0; j < layers.size(); ++j) {
            vector<T> newOutputs;
            for (int k = 0; k < layers[j].size(); ++k) {
                layers[j][k].typeActivation(activationTypes[j]);
                newOutputs.push_back(layers[j][k].feedForward(outputs.back()));
            }
            outputs.push_back(newOutputs);
        }

        // * calculate error (start from output layer and move backwards)
        for (int j = layers.size() - 1; j >= 0; --j) {
            vector<T> newErrors(layers[j].size(), 0);
            for (int k = 0; k < layers[j].size(); ++k) {
                T error = 0.0;
                if (j == layers.size() - 1) {  // output layer
                    error = outputLayerError(outputs[j + 1][k], targets[i][k]);
                } else {  // hidden layers
                    for (int l = 0; l < layers[j + 1].size(); ++l) {
                        // * use output error for hidden layer
                        error += hiddenLayerError(outputs[j + 1][l], newErrors[l], layers[j + 1][l].getWeights()[k]);
                    }
                }
                newErrors[k] = error;
            }

            // * update weights and bias
            for (int k = 0; k < layers[j].size(); ++k) {
                for (int l = 0; l < layers[j][k].getWeights().size(); ++l) {
                    layers[j][k].setWeights(l, updateWeights(layers[j][k].getWeights()[l], learningRate, newErrors[k], outputs[j][l]));
                }
                layers[j][k].setBias(updateBias(layers[j][k].getBias(), learningRate, newErrors[k]));
            }
        }
    }
}

template <typename T>
T MultiLayerPerceptron<T>::updateWeights(const T weight, const T learningRate, const T error, const T input) {
    return weight + (learningRate * error * input);
}

template <typename T>
T MultiLayerPerceptron<T>::updateBias(const T bias, const T learningRate, const T error) {
    return bias + (learningRate * error);
}

template <typename T>
T MultiLayerPerceptron<T>::hiddenLayerError(const T output, const T error, const T weight) {
    return output * (1 - output) * error * weight;
}

template <typename T>
T MultiLayerPerceptron<T>::outputLayerError(const T output, const T target) {
    return output * (1 - output) * (target - output);
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
            T error = errors[j] * activationDerivative(layerOutputs[i + 1][j], activationTypes[i]);
            for (int k = 0; k < layers[i][j].weights.size(); ++k) {
                layers[i][j].weights[k] += error * layerOutputs[i][k] * learningRate;
                #pragma omp atomic
                newErrors[k] += layers[i][j].weights[k] * error;
            }
            layers[i][j].bias += error * learningRate;
        }

        errors = newErrors;
    }

    // // * display inputs and outputs
    // cout << "Inputs: ";
    // for (int i = 0; i < inputs.size(); ++i) {
    //     cout << inputs[i] << " ";
    // }
    // cout << "Outputs: ";
    // for (int i = 0; i < outputs.size(); ++i) {
    //     cout << outputs[i] << " ";
    // }
    // cout << endl;
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
T MultiLayerPerceptron<T>::activationDerivative(T x, string type)
{

    // * change type to lower case
    for (int i = 0; i < type.length(); i++)
    {
        type[i] = tolower(type[i]);
    }
    
    if (type == "linear")
    {
        // * f(x) = x
        return 1;
    }
    else if (type == "sigmoid")
    {
        // * f(x) = 1 / (1 + e^(-x))
        return x * (1 - x);
    }
    else if (type == "tanh")
    {
        // * f(x) = tanh(x)
        return 1 - x * x;
    }
    else if (type == "relu")
    {
        // * f(x) = max(0, x)
        return x > 0 ? 1 : 0;
    }
    else if (type == "leakyrelu")
    {
        // * f(x) = max(0.01x, x)
        return x > 0 ? 1 : 0.01;
    }
    else if (type == "softmax")
    {
        // * f(x) = e^x / sum(e^x)
        return x * (1 - x);
    }
    else if (type == "step")
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
            if (abs(targets[i][j] - output[j]) < accuracy) {
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
            if (abs(targets[i][j] - output[j]) > accuracy) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
void MultiLayerPerceptron<T>::setActivation(const vector<string> &activationTypes)
{
    this->activationTypes = activationTypes;
    for (int i = 0; i < layers.size(); ++i) {
        for (int j = 0; j < layers[i].size(); ++j) {
            layers[i][j].typeActivation(activationTypes[i]);
        }
    }
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
void MultiLayerPerceptron<T>::setAccuracy(T accuracy)
{
    this->accuracy = accuracy;
}

template <typename T>
void MultiLayerPerceptron<T>::display()
{
    // * Display layers
    for (int i = 0; i < layers.size(); ++i) {
        cout << "Layer: " << i << " -> " << "activation: " << activationTypes[i] << endl;
        cout << "Base accuracy: " << accuracy << endl;
        for (int j = 0; j < layers[i].size(); ++j) {
            cout << "Node: " << j << " ";
            // * display weights and bias
            cout << "W: ";
            for (int k = 0; k < layers[i][j].getWeights().size(); ++k) {
                cout << layers[i][j].getWeights()[k] << " ";
            }
            cout << "B: " << layers[i][j].getBias() << endl;
        }
    }
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
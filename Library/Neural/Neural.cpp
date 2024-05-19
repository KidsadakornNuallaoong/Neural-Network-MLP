#include "Neural.h"

using namespace std;
using std::fixed;
using std::vector;

template <typename T>
Neural<T>::Neural()
{
    // * default constructor
    this->inputs = {};
    this->weights = {};
}

template <typename T>
Neural<T>::~Neural()
{
    // * default destructor
    this->inputs.clear();
    this->weights.clear();
}

template <typename T>
void Neural<T>::setInputs(vector<T> inputs, bool verbose)
{
    this->inputs = inputs;
    if(verbose){
        cout << "Inputs Size: " << inputs.size() << endl;
        cout << "Inputs :  ";
        for (int i = 0; i < inputs.size(); i++)
        {
            cout << inputs[i] << " ";
        }
        cout << endl;
    }
}

template <typename T>
void Neural<T>::setWeights(vector<vector<T>> weights, bool verbose)
{
    // * protection error
    if (weights.size() > this->inputs.size())
    {
        cout << "\033[1;31m" << "Error: Weights size is greater than inputs size" << "\033[0m" << endl;
        return;
    }
    
    // * set weights
    this->weights = weights;
    // * display weights
    if(verbose){
        cout << "Weights Size: " << weights.size() << endl;
        for (int i = 0; i < weights.size(); i++)
        {
            cout << "Weights[" << i << "] Size :  " << weights[i].size() << endl;
            cout << "Weights[" << i << "] :  ";
            for (int j = 0; j < weights[i].size(); j++)
            {
                cout << weights[i][j] << " ";
            }
            cout << endl;
        }
    }
}

template <typename T>
void Neural<T>::setBiasWeight(vector<vector<T>> biasWeights, bool verbose)
{
    // * protection error
    if (biasWeights.size() > this->inputs.size())
    {
        cout << "\033[1;31m" << "Error: Bias Weights size is greater than inputs size" << "\033[0m" << endl;
        return;
    }
    
    // * set bias weights
    this->biasWeight = biasWeights;
    // * display bias weights
    if(verbose){
        cout << "Bias Weights Size: " << biasWeights.size() << endl;
        for (int i = 0; i < biasWeights.size(); i++)
        {
            cout << "Bias Weights[" << i << "] Size :  " << biasWeights[i].size() << endl;
            cout << "Bias Weights[" << i << "] :  ";
            for (int j = 0; j < biasWeights[i].size(); j++)
            {
                cout << biasWeights[i][j] << " ";
            }
            cout << endl;
        }
    }
}

template <typename T>
void Neural<T>::MultiLayerPerceptron(int layer)
{
    // * Layer
    for(int i = 0; i < layer; i++){
        // * set perceptrons
        for (int j = 0; j < this->weights.size(); j++)
        {
            Perceptron<T> p(this->inputs, this->weights[j]);
            this->perceptrons[i].push_back(p);
            this->hiddenLayer.hiddenNeurons++;
        }
        this->hiddenLayer.hiddenCount++;
    }
    for(int i = 0; i < this->perceptrons.size(); i++){
        cout << "Perceptrons[" << i << "] Size: " << this->perceptrons[i].size() << endl;
        for(int j = 0; j < this->perceptrons[i].size(); j++){
            cout << "Perceptrons[" << i << "][" << j << "] : " << endl;
            this->perceptrons[i][j].display();
        }
    }
}

// template <typename T>
// void Neural<T>::setHiddenLayer()
// {
//     int N = 0;
//     if(this->inputs.size() > this->weights.size()){
//         N = this->inputs.size();
//     } else {
//         N = this->weights.size();
//     }
//     // * set perceptrons
//     for (int i = 0; i < this->weights.size(); i++)
//     {
//         vector<T> w;
//         for (int j = 0; j < N; j++)
//         {
//             w.push_back(this->weights[i][j%this->weights[i].size()]);
//         }
//         Perceptron<T> p(this->inputs, w);
//         this->perceptrons.push_back(p);
//         this->hiddenLayer.hiddenNeurons++;
//     }
//     this->hiddenLayer.hiddenCount++;
// }

template <typename T>
void Neural<T>::display(string option)
{
    // * change option to uppercase
    for(int i = 0; i < option.size(); i++){
        option[i] = toupper(option[i]);
    }
    if(option == "NEURON"){
        this->displayNeurons();
    } else if(option == "PERCEPTRON"){
        this->displayPerceptrons();
    } else {
        this->displayNeurons();
    }
}

template <typename T>
void Neural<T>::displayNeurons()
{
    cout << "Neural Network" << endl;
    cout << "Inputs Size: " << this->inputs.size() << endl;
    cout << "Inputs :  ";
    for (int i = 0; i < this->inputs.size(); i++)
    {
        cout << this->inputs[i] << " ";
    }
    cout << endl;
    cout << "Weights Size: " << this->weights.size() << endl;
    for (int i = 0; i < this->weights.size(); i++)
    {
        cout << "Weights[" << i << "] Size [" << this->weights[i].size() << "] :  ";
        for (int j = 0; j < this->weights[i].size(); j++)
        {
            cout << this->weights[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Bias Weights Size: " << this->biasWeight.size() << endl;
    for (int i = 0; i < this->biasWeight.size(); i++)
    {
        cout << "Bias Weights[" << i << "] Size [" << this->biasWeight[i].size() << "] :  ";
        for (int j = 0; j < this->biasWeight[i].size(); j++)
        {
            cout << this->biasWeight[i][j] << " ";
        }
        cout << endl;
    }
}

template <typename T>
void Neural<T>::displayPerceptrons()
{
}

template class Neural<float>;
template class Neural<double>;
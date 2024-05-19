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
void Neural<T>::setHiddenLayer()
{
    int N = 0;
    if(this->inputs.size() > this->weights.size()){
        N = this->inputs.size();
    } else {
        N = this->weights.size();
    }
    // * set perceptrons
    for (int i = 0; i < this->weights.size(); i++)
    {
        vector<T> w;
        for (int j = 0; j < N; j++)
        {
            w.push_back(this->weights[i][j%this->weights[i].size()]);
        }
        Perceptron<T> p(this->inputs, w);
        this->perceptrons.push_back(p);
    }
}

template <typename T>
void Neural<T>::display()
{
    // * display : inputs i : ? | weights i : ? j : ?
    // cout << "Inputs Size: " << this->inputs.size() << endl;
    // for(int i = 0; i < this->inputs.size(); i++){
    //     cout << "Inputs[" << i << "] : " << this->inputs[i] << endl;
    // }

    // * display : perceptrons
    cout << "Hiddens Size: " << this->perceptrons.size() << endl;
    for (int i = 0; i < this->perceptrons.size(); i++)
    {
        cout << "Hidden[" << i << "] :  " << endl;
        this->perceptrons[i].display();
    }
}

template class Neural<float>;
template class Neural<double>;
#if !defined(NEURON_H)
#define NEURON_H

#pragma once
#include <iostream>
#include <vector>

class neuron {
    public:
        neuron(){};
        ~neuron(){};

        double initNeuron(){
            return ((double)rand() / RAND_MAX);
        };
    
    protected:
        std::vector<double> listOfWeightsIn;
        std::vector<double> listOfWeightsOut;
        double outputValue;
        double error;
        double sensibility;
        
};


#endif // NEURON_H

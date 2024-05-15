#if !defined(LAYER_H)
#define LAYER_H

#pragma once
#include <iostream>
#include <vector>
#include "neuron.h"

class layer {
    public:
        layer(){};
        ~layer(){};

        void printLayer(){
            std::cout << "Layer" << std::endl;
        };

    protected:
        std::vector<neuron> listOfNeurons;
        size_t numberOfNeuronInLayer;
};

#endif // LAYER_H

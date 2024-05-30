#include <iostream>
#include <cmath>
#include <vector>

#include "../Library/Perceptron/Perceptron.h"
#include "../Library/Neural/Neural.h"
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

int main() {
    // int epoch = 10000;
    // double acc = 1e-6;
    
    // Perceptron<double> p;
    // p.setInputs({1});
    // p.setWeights({0.1});
    // p.setBias(0.1);
    // p.setActivation("Linear");
    // p.setTarget(2);
    // p.setLearningRate(0.001);
    // p.feedForward();
    // p.display();


    // cout << endl;
    // p.train(epoch, acc);

    // // plt::ion();
    // vector<double> x_data, y_data;
    // for (int i = 0; i < 1000; i++) {
    //     p.backpropagate();
    //     // x_data.push_back(i);
    //     // y_data.push_back(p.MSE());
    //     // plt::cla();
    //     // plt::plot(x_data, y_data, "r-");
    //     // plt::pause(0.0001);
    // }


    // p.display();
    // cout << endl;

    // Perceptron<double> p2(&p);
    // // p2.copyEnv(&p);
    // for(int i = 0; i < 1000; i++) {
    //     p2.setInputs({double(i)});
    //     p2.setTarget(i+1);
    //     p2.feedForward();
    //     p2.display();
    //     // x_data.push_back(i);
    //     // y_data.push_back(p2.getOutput());
    //     // plt::plot(x_data, x_data, "r-");
    //     // plt::plot(x_data, y_data, "g-");
    //     // plt::pause(0.01);
    // }

    // * try multi layer perceptron
    Perceptron<double> p1;
    vector<double> input1 = {1, 2};
    p1.setInputs(input1);
    p1.setWeights({0.1, 0.1});
    p1.setBias(0.1);
    p1.setActivation("Linear");
    p1.setLearningRate(0.001);
    p1.feedForward();
    p1.display();

    cout << endl;

    Perceptron<double> p2;
    p2.setInputs(input1);
    p2.setWeights({0.2, 0.2});
    p2.setBias(0.1);
    p2.setActivation("Linear");
    p2.setLearningRate(0.001);
    p2.feedForward();
    p2.display();

    cout << endl;

    Perceptron<double> p3;
    p3.setInputs({p1.getOutput(), p2.getOutput()});
    p3.setWeights({0.5, 0.5});
    p3.setBias(0.1);
    p3.setTarget(3);
    p3.setActivation("Linear");
    p3.setLearningRate(0.001);
    p3.feedForward();

    cout << endl;

    // * backpropagate
    // * δ = (y - y') * f'(z)
    cout << "Backpropagate p3: " << p3.getOutput() << endl;
    cout << "Delta: " << p3.Err() * p3.getOutput() << endl;

    // * epoch multi layer perceptron
    plt::ion();
    vector<double> x_data, x1_data, x2_data, y_data;
    for (int i = 0; i < 1200; i++) {
        if(p3.Err() < 1e-6 && p3.Err() > -1e-6) {
            break;
        }
        p3.setInputs({p1.getOutput(), p2.getOutput()});

        // * update weights
        // * δ = err * W * f'(z)
        p1.setError(p3.Err() * p3.getWeights()[0] * p1.getOutput());
        p2.setError(p3.Err() * p3.getWeights()[1] * p2.getOutput());
        p1.backpropagate();
        p2.backpropagate();
        p3.backpropagate();
        
        cout << "p1 output: " << p1.getOutput() << " p2 output: " << p2.getOutput() << endl;
        cout << "p3 output: " << p3.getOutput() << " p3 error: " << p3.Err() << endl << endl;

        // * plot
        x_data.push_back(i);
        y_data.push_back(p3.MSE());

        plt::cla();
        plt::plot(x_data, y_data, "r-");
        plt::pause(0.0001);
    }
    
    p1.display();
    cout << endl;

    p2.display();
    cout << endl;

    p3.display();
    cout << endl;

    // * predict
    // for(int i = 0; i < 100; i++) {
    //     vector<double> input = {double(i), double(i+1)};
    //     Perceptron<double> N1(&p1);
    //     Perceptron<double> N2(&p2);
    //     N1.setInputs(input);
    //     N2.setInputs(input);
    //     N1.feedForward();
    //     N2.feedForward();

    //     Perceptron<double> N3(&p3);
    //     N3.setInputs({N1.getOutput(), N2.getOutput()});
    //     N3.setTarget(input[input.size()-1] + 1);
    //     N3.feedForward();
    //     N3.display();

    //     x_data.push_back(i);
    //     x1_data.push_back(input[0]);
    //     x2_data.push_back(input[1]);
    //     y_data.push_back(N3.getOutput());

    //     plt::cla();
    //     plt::plot(x_data, x1_data, "r*");
    //     plt::plot(x_data, x2_data, "g*");
    //     plt::plot(x_data, y_data, "g-");
    //     plt::pause(0.01);
    // }
    return 0;
}
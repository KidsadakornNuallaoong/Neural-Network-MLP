#include <iostream>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>

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

//  !! Neural Network
    // // * try multi layer perceptron
    // Perceptron<double> p1;
    // vector<double> input1 = {3, 4};
    // double LR = 1e-4;
    // p1.setInputs(input1);
    // p1.setWeights({0.1, 0.2});
    // p1.setBias(0.2);
    // p1.setActivation("Linear");
    // p1.setLearningRate(LR);
    // p1.feedForward();
    // p1.display();

    // cout << endl;

    // Perceptron<double> p2;
    // p2.setInputs(input1);
    // p2.setWeights({0.2, 0.3});
    // p2.setBias(-0.2);
    // p2.setActivation("Linear");
    // p2.setLearningRate(LR);
    // p2.feedForward();
    // p2.display();

    // cout << endl;

    // Perceptron<double> p3;
    // p3.setInputs({p1.getOutput(), p2.getOutput()});
    // p3.setWeights({0.1, 0.1});
    // p3.setBias(0.1);
    // p3.setTarget(5);
    // p3.setActivation("Linear");
    // p3.setLearningRate(LR);
    // p3.feedForward();

    // cout << endl;

    // // * backpropagate
    // // * δ = (y - y') * f'(z)
    // cout << "Backpropagate p3: " << p3.getOutput() << endl;
    // cout << "Delta: " << p3.Err() * p3.getOutput() << endl;

    // // * epoch multi layer perceptron
    // plt::ion();
    // for (int i = 0; i < 1200; i++) {
    //     if(p3.Err() < 1e-6 && p3.Err() > -1e-6) {
    //         break;
    //     }
    //     p3.setInputs({p1.getOutput(), p2.getOutput()});

    //     // * update weights
    //     // * δ = err * W * f'(z)
    //     p1.setError(p3.Err() * p3.getWeights()[0] * p1.getOutput());
    //     p2.setError(p3.Err() * p3.getWeights()[1] * p2.getOutput());
    //     p1.backpropagate();
    //     p2.backpropagate();
    //     p3.backpropagate();
        
    //     cout << "p1 output: " << p1.getOutput() << " p2 output: " << p2.getOutput() << endl;
    //     cout << "p3 output: " << p3.getOutput() << " p3 error: " << p3.Err() << endl << endl;

    //     // * plot
    //     // x_data.push_back(i);
    //     // y_data.push_back(p3.MSE());

    //     // plt::cla();
    //     // plt::plot(x_data, y_data, "r-");
    //     // plt::pause(0.0001);
    // }
    
    // p1.display();
    // cout << endl;

    // p2.display();
    // cout << endl;

    // p3.display();
    // cout << endl;

    // // * predict
    // vector<double> x_data, x1_data, x2_data, y_data;
    // for(int i = 0; i < 100; i++) {
    //     vector<double> input = {double(i), double(i*2.2)};
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
        
    //     cout << endl;

    //     x_data.push_back(i);
    //     x1_data.push_back(input[0]);
    //     x2_data.push_back(input[1]);
    //     y_data.push_back(N3.MSE());

    //     plt::cla();
    //     plt::plot(x_data, x2_data, "g*");
    //     plt::plot(x_data, y_data, "g-");
    //     plt::pause(0.01);
    // }

    // * generate data
    vector<double> dat_a, dat_b, w;
    // cout << "Data: " << endl;
    for (int i = 0; i < 100; i++) {
        double a = i*5;
        double b = sin(5 * a - 10 * (rand() % 100) + 0.4)/cos(5 * a - 10 * (rand() % 100) + 0.4);
        dat_a.push_back(a);
        dat_b.push_back(b);
        w.push_back(0.1);

        // plt::plot(dat_a, dat_b, "r*");
        // plt::pause(0.01);
        // plt::show();
    }


    // // * train model
    // plt::ion();
    // Perceptron<double> p[dat_a.size()];
    // vector<double> x_data, y_data, z_data;
    // for (int i = 0; i < dat_a.size(); i++) {
    //     p[i].setInputs({dat_a[i]});
    //     p[i].setWeights({w[i]});
    //     p[i].setBias(0.1);
    //     p[i].setActivation("Linear");
    //     p[i].setLearningRate(1e-6);
    //     p[i].setTarget(dat_b[i]);
    //     p[i].feedForward();

    //     // * train
    //     for (int j = 0; j < 1000; j++) {
    //     z_data.push_back(p[i].getOutput());
    //         p[i].backpropagate();
    //         if (p[i].MSE() < 1e-6 && p[i].MSE() > -1e-6) {
    //             break;
    //         }
    //     }

    //     cout << "Data fitting: " << i << " MSE: " << p[i].MSE() << endl << endl;
    //     cout << "Target: " << dat_b[i] << endl;
    //     cout << "Output: " << p[i].getOutput() << endl << endl;
    // }
    
    // x_data.push_back(dat_a[i]);
    // y_data.push_back(dat_b[i]);
    // z_data.push_back(p[i].getOutput());
    // plt::plot(x_data, y_data, "r*");
    // plt::plot(x_data, z_data, "g-");
    // plt::show();

    // * train model
    plt::ion();
    Perceptron<double> p[dat_a.size()];
    vector<double> out;
    vector<double> x_data, y_data, z_data;
    for (int i = 0; i < dat_a.size(); i++) {
        p[i].setInputs({dat_a[i]});
        p[i].setWeights({w[i]});
        p[i].setBias(0.1);
        p[i].setActivation("Linear");
        p[i].setLearningRate(1e-6);
        p[i].setTarget(dat_b[i]);
        p[i].feedForward();

        out.push_back(p[i].getOutput());
    }

    cout << "Output: " << endl;
    for (int i = 0; i < out.size(); i++) {
        cout << "Output: " << out[i] << " Target: " << dat_b[i] << endl;
    }

    // * train
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < dat_a.size(); j++) {
            p[j].backpropagate();
        }

        // * find max and min mse
        double mse = 0;
        for (int j = 0; j < dat_a.size(); j++) {
            mse += p[j].MSE();
        }
        mse /= dat_a.size();
        cout << "Data fitting: " << i << " MSE: " << mse << endl << endl;

        plt::cla();
        plt::plot(dat_a, dat_b, "r*");
        vector<double> y_data;
        for (int j = 0; j < dat_a.size(); j++) {
            // * + curve
            y_data.push_back(p[j].getOutput());
        }
        plt::plot(dat_a, y_data, "g.-");
        plt::pause(0.01);

        // * break if mse is small
        if (mse < 1e-2 && mse > -1e-2) {
            break;
        }
    }

    // // * train
    // Perceptron<double> p[dat_a.size()];
    // for (int i = 0; i < dat_a.size(); i++) {
    //     p[i].setInputs({dat_a[i]});
    //     p[i].setWeights({w[i]});
    //     p[i].setBias(0.1);
    //     p[i].setActivation("Linear");
    //     p[i].setLearningRate(1e-6);
    //     p[i].setTarget(dat_b[i]);
    //     p[i].feedForward();
    // }

    // // * train
    // double mse = 0;
    // double max_mse, min_mse;
    // plt::figure_size(1200, 800);
    // vector<double> x, y;
    // vector<std::thread> threads;
    // mutex m;
    // for (int i = 0; i < 19000; i++) {
    //     for (int j = 0; j < dat_a.size(); j++) {
    //         p[j].backpropagate();
    //     }

    //     // * find max and min mse
    //     for (int j = 0; j < dat_a.size(); j++) {
    //         mse += p[j].MSE();
    //     }
    //     mse /= dat_a.size();
    //     cout << "Data fitting: " << i << " MSE: " << mse << endl << endl;
    //     // cout << "Max MSE: " << max_mse << endl;
    //     // cout << "Min MSE: " << min_mse << endl;
    //     // cout << "MSE:" << (max_mse - min_mse) << endl;

    //     plt::cla();
    //     // plt::plot(dat_a, dat_b, "r*");
    //     plt::scatter_colored(dat_a, dat_b, w, 50);
    //     vector<double> y_data;
    //     for (int j = 0; j < dat_a.size(); j++) {
    //         // * + curve
    //         y_data.push_back(p[j].getOutput());
    //     }
    //     plt::plot(dat_a, y_data, "g.-");
    //     plt::pause(0.01);

    //     // * plot mse in new window
    //     // x.push_back(i);
    //     // y.push_back(mse);
    //     // plt::cla();
    //     // plt::plot(x, y, "r-");
    //     // plt::pause(0.01);

    //     // * break if mse is small
    //     if (mse < 1e-2 && mse > -1e-2) {
    //         break;
    //     }
    // }
    return 0;
}
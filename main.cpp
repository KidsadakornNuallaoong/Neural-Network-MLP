#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// #include "../Library/Perceptron/Perceptron.hpp"
#include "./Library/MLP/MLP.hpp"

#ifdef _WIN32
    #include <thread>
    #include <conio.h> // For _kbhit and _getch
    
    bool running = true;

    void checkInput() {
        while (running) {
            if (_kbhit()) {
                char ch = _getch();
                if (ch == 'q' || ch == 'Q') {
                    running = false;
                }
            }
        }
    }

#elif __linux__
    #include <thread>
    #include <atomic>
    #include <unistd.h>
    #include <fcntl.h>
    #include <termios.h>

    std::atomic<bool> running(true);

    bool kbhit() {
        struct termios oldt, newt;
        int ch;
        int oldf;

        // Get the current terminal settings
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        // Disable canonical mode and echo
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        // Set stdin to non-blocking mode
        oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

        // Check for input
        ch = getchar();

        // Restore terminal settings
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldf);

        if(ch != EOF) {
            ungetc(ch, stdin);
            return true;
        }

        return false;
    }

    void checkInput() {
        while (running) {
            if (kbhit()) {
                char ch = getchar();
                if (ch == 'q' || ch == 'Q') {
                    running = false;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small delay to avoid high CPU usage
        }
    }
#endif

int main() {

    // * Test Multi-Layer Perceptron
    vector<int> layersSize = {2, 3, 1};
    MultiLayerPerceptron<double> mlp(layersSize);
    
    // * XOR Problem
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};
    mlp.setLayerWeights(0, {{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}});
    mlp.setLayerBias(0, {0.5, 0.5, 0.5});
    mlp.setLayerWeights(1, {{0.5, 0.5, 0.5}});
    mlp.setLayerBias(1, {0.5});
    mlp.setActivation({"sigmoid", "sigmoid"});
    mlp.setAccuracy(0.01);
    mlp.display();
    double learningRate = 0.1;
    int iterations = 0;


    for(const auto& input : inputs) {
        cout << "Input: ";
        for(const auto& i : input) {
            cout << i << " ";
        }
        cout << "Output: " << mlp.feedForward(input)[0] << endl;
    }

    cout << "Training..." << endl;
    // while (!mlp.allOutputsCorrect(inputs, targets)) {
    //     int index = rand() % inputs.size();
    //     mlp.train(inputs[index], targets[index], learningRate);
    //     iterations++;

    //     cout << "accuracy: " << mlp.calculateAccuracy(inputs, targets) << " loss: " << mlp.calculateLoss(inputs, targets) << endl;
    // }

    // for(int i = 0; i < 2000000; i++) {
    //     mlp.backPropagation(inputs, targets, learningRate);
    //     iterations++;
    //     i++;

    //     cout << "accuracy: " << mlp.calculateAccuracy(inputs, targets) << " loss: " << mlp.calculateLoss(inputs, targets) << endl;
    // }

    std::thread inputThread(checkInput);

    while (running) {
        mlp.backPropagation(inputs, targets, learningRate);
        iterations++;
        cout << "Iterations: " << iterations << " Accuracy: " << mlp.calculateAccuracy(inputs, targets) * 100 << "%" << " Loss: " << mlp.calculateLoss(inputs, targets) << endl;
    }

    inputThread.join();
    
    // * alert when training finished
    cout << endl;
    cout << "Training finished!" << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Accuracy: " << mlp.calculateAccuracy(inputs, targets) * 100 << "%" << endl;
    cout << "Loss: " << mlp.calculateLoss(inputs, targets) << endl;
    cout << "All outputs correct: " << mlp.allOutputsCorrect(inputs, targets) << endl;
    cout << endl;

    mlp.display();
    for(const auto& input : inputs) {
        cout << "Input: ";
        for(const auto& i : input) {
            cout << i << " ";
        }
        cout << "Output: " << mlp.feedForward(input)[0] << endl;
    }

    // mlp.display();

    return 0;
}

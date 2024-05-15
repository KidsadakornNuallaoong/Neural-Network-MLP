#include <iostream>
#include <cmath>

using namespace std;

int main(){
    // * input layer
    float input[3][3] = {
        {0, 0, 1},
        {1, 1, 0},
        {1, 0, 1}
    };

    float weights[3][3] = {
        {0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5},
        {0.5, 0.5, 0.5}
    };
    float bias = 0;

    float hidden[3];
    hidden[0] = input[0][0] * weights[0][0] + input[0][1] * weights[0][1] + input[0][2] * weights[0][2] + bias;
    hidden[1] = input[1][0] * weights[1][0] + input[1][1] * weights[1][1] + input[1][2] * weights[1][2] + bias;
    hidden[2] = input[2][0] * weights[2][0] + input[2][1] * weights[2][1] + input[2][2] * weights[2][2] + bias;

    // * output
    float output[3];
    output[0] = 1 / (1 + exp(-hidden[0]));
    output[1] = 1 / (1 + exp(-hidden[1]));
    output[2] = 1 / (1 + exp(-hidden[2]));

    cout << "Output 1 : " << output[0] << endl;
    cout << "Output 2 : " << output[1] << endl;
    cout << "Output 3 : " << output[2] << endl;
}
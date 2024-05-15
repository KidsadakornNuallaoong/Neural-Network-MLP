#include <iostream>
#include <cmath>

using namespace std;

double hiddenFunction(double input, double weight) {
    return input * weight;
}

double hiddenFunctionArray(double* input, double* weight, double bias, int size, bool display = false) {
    double sum = 0;
    for(int i = 0; i < size; i++) {
        sum += hiddenFunction(input[i], weight[i%(size-1)]);
        if(display) {
            cout << "size: " << size;
            cout << " input index: " << i;
            cout << " weight index: " <<  i%(size-1) << endl;
            cout << "input = " << input[i] << " weight = " << weight[i%(size-1)] << " sum = " << sum << endl << endl;
        }
    }
    sum += bias;
    return sum;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

int main() {
    // * x1 = 15, x2 = 30
    // * formula x^2
    double input[] = {(15*15), (30*30), (30*30)};

    // * weight
    // * Object you need to find
    double weight[] = {27, 50};

    // * object or number you need
    double b = -50000;

    // * auto calculate the size of the array
    int size = sizeof(input) / sizeof(double); // Calculate the size of the array
    double a = sigmoid(hiddenFunctionArray(input, weight, b, size));

    if(a > 0) {
        cout << "1" << endl;
    } else {
        cout << "0" << endl;
    }

    printf("a = %f\n", a);

    return 0;
}
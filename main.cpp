#include <iostream>
#include <vector>

using namespace std;

// #include "../Library/Perceptron/Perceptron.hpp"
#include "./Library/MLP/MLP.hpp"

vector<int> text_to_binary(string text){
    vector<int> binary;
    for (int i = 0; i < int(text.size()); i++){
        for (int j = 0; j < 8; j++){
            binary.push_back((text[i] >> j) & 1);
        }
    }
    return binary;
}

string binary_to_text(vector<int> binary){
    string text = "";
    for (int i = 0; i < int(binary.size()); i += 8){
        int c = 0;
        for (int j = 0; j < 8; j++){
            c += binary[i + j] << j;
        }
        text += c;
    }
    return text;
}

int count_max_size(vector<vector<int>> binary_texts){
    int max_size = 0;
    for (int i = 0; i < int(binary_texts.size()); i++){
        if (int(binary_texts[i].size()) > max_size){
            max_size = int(binary_texts[i].size());
        }
    }
    return max_size;
}

vector<int> padding(vector<int> binary_text, int max_size){
    vector<int> padded_text = binary_text;
    for (int i = 0; i < max_size - int(binary_text.size()); i++){
        padded_text.push_back(0);
    }
    return padded_text;
}

vector<int> double_to_int(vector<double> double_vector){
    vector<int> int_vector;
    for (int i = 0; i < int(double_vector.size()); i++){
        int_vector.push_back(round(double_vector[i]));
    }
    return int_vector;
}

vector<double> int_to_double(vector<int> int_vector){
    vector<double> double_vector;
    for (int i = 0; i < int(int_vector.size()); i++){
        double_vector.push_back(double(int_vector[i]));
    }
    return double_vector;
}

int main() {

    vector<string> texts = {"zero", "zero", "one", "two"};

    vector<vector<int>> binary_texts;

    for (int i = 0; i < int(texts.size()); i++){
        binary_texts.push_back(text_to_binary(texts[i]));
    }

    // * padding
    int max_size = count_max_size(binary_texts);
    for (int i = 0; i < int(binary_texts.size()); i++){
        binary_texts[i] = padding(binary_texts[i], max_size);
    }

    // * convert binary to text
    // for (int i = 0; i < int(binary_texts.size()); i++){
    //     cout <<  << endl;
    // }

    // * display binary texts
    // for (int i = 0; i < int(binary_texts.size()); i++){
    //     for (int j = 0; j < int(binary_texts[i].size()); j++){
    //         cout << binary_texts[i][j];
    //     }
    //     cout << endl;
    // }
    
    // * display binary texts
    // for (int i = 0; i < int(binary_texts.size()); i++){
    //     for (int j = 0; j < int(binary_texts[i].size()); j++){
    //         cout << binary_texts[i][j];
    //     }
    //     cout << endl;
    // }

    // * change vector int to vector double
    vector<vector<double>> binary_texts_double;
    for (int i = 0; i < int(binary_texts.size()); i++){
        binary_texts_double.push_back(int_to_double(binary_texts[i]));
    }

    // * display binary texts double
    // for (int i = 0; i < int(binary_texts_double.size()); i++){
    //     for (int j = 0; j < int(binary_texts_double[i].size()); j++){
    //         cout << binary_texts_double[i][j];
    //     }
    //     cout << endl;
    // }
    
    // * 0, 1, 2, 3, 4
    vector<vector<double>> img = {
        {0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0},

         {0, 0, 0, 1, 1, 0, 0, 0,
          0, 0, 1, 1, 1, 1, 0, 0,
          0, 1, 1, 0, 0, 1, 1, 0,
          0, 1, 1, 0, 0, 1, 1, 0,
          0, 1, 1, 0, 0, 1, 1, 0,
          0, 1, 1, 0, 0, 1, 1, 0,
          0, 0, 1, 1, 1, 1, 0, 0,
          0, 0, 0, 1, 1, 0, 0, 0},
         
         {0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 1, 1, 0, 0, 0,
          0, 0, 1, 1, 1, 0, 0, 0,
          0, 0, 0, 1, 1, 0, 0, 0,
          0, 0, 0, 1, 1, 0, 0, 0,
          0, 0, 1, 1, 1, 1, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0},

         {0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 1, 1, 0, 0, 0,
          0, 0, 1, 1, 1, 1, 0, 0,
          0, 0, 0, 0, 1, 1, 0, 0,
          0, 0, 0, 1, 1, 0, 0, 0,
          0, 0, 1, 1, 0, 0, 0, 0,
          0, 0, 1, 1, 1, 1, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0},
    };

    // * XOR Problem
    // vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> inputs = img;
    // vector<vector<double>> targets = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    vector<vector<double>> targets = {
        binary_texts_double[0],
        binary_texts_double[1],
        binary_texts_double[2],
        binary_texts_double[3]
    };

    cout << "Max size: " << count_max_size(binary_texts) << endl;

    // * Test Multi-Layer Perceptron
    vector<int> layersSize = {8*8, 64, 32, 32, max_size};
    MultiLayerPerceptron<double> mlp(layersSize);

    mlp.setActivation({"sigmoid", "sigmoid", "sigmoid", "sigmoid"});
    mlp.setAccuracy(0.05);
    // mlp.display();

    double learningRate = 0.1;

    // cout << "Initial outputs:" << endl;
    mlp.predict(inputs, NONE);

    // cout << "Training..." << endl;

    mlp.train(inputs, targets, learningRate);


    // cout << "Initial outputs:" << endl;
    vector<vector<double>> inputs_test = {
        {0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0},

         {0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 0, 0.5, 0,
          0, 0, 0, 1, 1, 0, 0, 0,
          0, 0.5, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 1, 0, 1, 0, 0, 0,
          0, 0.2, 0, 1, 1, 1, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0},

        {0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 1, 1, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0},
    };

    vector<vector<double>> result = mlp.predict(inputs_test, R_D);

    // * convert binary to text
    for (int i = 0; i < int(result.size()); i++){
        cout << "Answer: " << binary_to_text(double_to_int(result[i])) << endl;
    }

    // if(round(result[0][0])){
    //     cout << "Number predict: " << "0" << " rate: " << result[0][0] << endl;
    // } else if(round(result[0][1])){
    //     cout << "Number predict: " << "1" << " rate: " << result[0][1] << endl;
    // } else if(round(result[0][2])){
    //     cout << "Number predict: " << "2" << " rate: " << result[0][2] << endl;
    // }

    // mlp.display();
    // mlp.export_to_json("model.json");
    // cout << "Result: " << mlp.predict(inputs_test, DISPLAY) << endl;

    return 0;
}

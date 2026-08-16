// #include <iostream>
#include <fstream>
#include "nn.hpp"
#include <Eigen/Dense>
#include "reading_funcs.hpp"


int main() {
    Eigen::MatrixXd A0;
    
    read_input(A0);
    
    NeuralNetwork nn(0.5, A0);

    nn.add_layer(784, 16);
    nn.add_layer(16, 10);

    ParameterGetter p(nn,"params/W1.txt", "params/W2.txt", "params/b1.txt", "params/b2.txt");

    p.getParams();

    nn.run();

    nn.show_predicted_numb();

    return 0;
}   
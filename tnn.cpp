#include "nn.hpp"
#include <iostream>
#include <fstream>
#include <vector>

double TrainableNeuralNetwork::loss_func() const {
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        int numb = static_cast<int>(Y(0, i)); // Accesare corectă Eigen: rândul 0, coloana i
        sum += std::log(layers[1].A(numb, i));
    }
    return -sum / m;
}

void TrainableNeuralNetwork::forwardprop(){
    this->m = getA0().cols();
    //1.calcul Z, sume ponderate layer 1

    Eigen::MatrixXd Z1 = layers[0].W * getA0();
   
    //adaugarea biasului(pe coloane , ii vector)
    Z1 = Z1.colwise() + layers[0].b;
    layers[0].Z = Z1;

    Eigen::MatrixXd A1 = ReLU(Z1);

    layers[0].A = A1;

    //2.A2
    Eigen::MatrixXd Z2 = layers[1].W * layers[0].A;

    Z2 = Z2.colwise() + layers[1].b;
    layers[1].Z = Z2;

    Eigen::MatrixXd A2 = softmax(Z2);

    layers[1].A = A2;
}

void TrainableNeuralNetwork::backprop(){
    //?dZ for A2

    //one hot nums
    Eigen::MatrixXd one_hot_numbs(10, m);
    one_hot_numbs.setZero();

    for (int i = 0; i < m; i++) {
        int numb = static_cast<int>(Y(0, i)); // Accesare corectă Eigen
        one_hot_numbs(numb, i) = 1.0;
    }

    //dZ[2] = A[2] - Y 
    layers[1].dZ = layers[1].A - one_hot_numbs;

    //dW[2] = 1/m * dZ[2] * A[2]transposed
    layers[1].dW = 1.0/m * layers[1].dZ * layers[0].A.transpose();

    //dB[2] = 1/m * suma pe randuri (media aritm pt fiecare neuron)
    layers[1].db = 1.0/m * layers[1].dZ.rowwise().sum();



    //folosire .array() pt produs hadamard , element cu element , intre prima parant si rezultatele derivatei
    layers[0].dZ = ((layers[1].W.transpose() * layers[1].dZ).array() * ReLU_derivative(layers[0].Z).array()).matrix();

    layers[0].dW = 1.0/m * layers[0].dZ * getA0().transpose();

    layers[0].db = 1.0/m * layers[0].dZ.rowwise().sum();
}

void TrainableNeuralNetwork::update_params(){
    layers[0].W = layers[0].W - getLearningRate()*layers[0].dW;
    layers[0].b = layers[0].b - getLearningRate()*layers[0].db;

    layers[1].W = layers[1].W - getLearningRate()*layers[1].dW;
    layers[1].b = layers[1].b - getLearningRate()*layers[1].db;

}

void TrainableNeuralNetwork::train(int cycles){
    for(int i=0; i < cycles; i++){
        
        forwardprop();

        backprop();

        update_params();

        // every tenth cycle show progress
        if(i % 10 == 0){
            std::cout<<"~"<<"Training cycle: "<<i<<"~"<<'\n';
            show_accuracy();
        }
    }
}

void TrainableNeuralNetwork::show_accuracy()const{
    double correct = 0.0;

    for (int i = 0; i < m; i++) {
        int numb = static_cast<int>(Y(0, i)); // Accesare corectă Eigen
        Eigen::Index maxRow;
        
        layers[1].A.col(i).maxCoeff(&maxRow);
        
        if (maxRow == numb) {
            correct++;
        }
    }

    std::cout<<"-------------------------------------------------------------------------------------\n";
    std::cout<<"Correctly guessed digits: "<< static_cast<int>(correct)<<'\n';
    std::cout<<"Loss Function: "<<loss_func() << '\n';
    std::cout<<"Accuracy: "<< correct/m <<'\n';
    std::cout<<"-------------------------------------------------------------------------------------\n";

}


#pragma once
#include "Eigen/Dense"
#include <vector>
#include <string>
//! MatrixXd - matrice , VectorXd - matrice colana




class Layer{
public:
    Eigen::MatrixXd W;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

    // dZ și dW trebuie să fie MatrixXd, altfel backprop va crăpa
    Eigen::MatrixXd dZ;
    Eigen::MatrixXd dW;
    Eigen::VectorXd db;

public:
    Layer(int anterior_neurons, int curr_neurons){
        // Alocare și inițializare directă a dimensiunilor și valorilor
        W = Eigen::MatrixXd::Random(curr_neurons, anterior_neurons) * 0.001;
        b = Eigen::VectorXd::Zero(curr_neurons);
    }
};

class NeuralNetwork{


    //input layer, 784 X 1
    Eigen::MatrixXd A0;

    
    double learning_rate;

    public:

    std::vector<Layer> layers;

    NeuralNetwork(double l, Eigen::MatrixXd input_layer):learning_rate(l), A0(input_layer) {};

    void add_layer(int ant_neurons, int curr_neurons);

    Eigen::MatrixXd softmax(Eigen::MatrixXd Z2);

    Eigen::MatrixXd ReLU(Eigen::MatrixXd Z1);

    Eigen::MatrixXd ReLU_derivative(Eigen::MatrixXd Z);

    double getLearningRate() const;

    //returneaza copie
    Eigen::MatrixXd getA0();

    virtual void forwardprop();    

    void show_predicted_numb()const;

    void run();
};

class TrainableNeuralNetwork:public NeuralNetwork{
    Eigen::RowVectorXd Y;

    //number of photos in training matrix
    //ar trebui corectata ca deocamdata ii hardcofdata pt training set , ca mai intai o antrenezi dupa vezi ce face pe celalalt dataset
    int m;

    public:
    TrainableNeuralNetwork(double l, Eigen::MatrixXd input_layer, Eigen::RowVectorXd Y): NeuralNetwork(l, input_layer), Y(Y){}

    void backprop();//schimba dw, dz, db

    void update_params();

    void train(int cycles);

    void show_accuracy()const;

    void forwardprop() override;
     
    double loss_func()const;

};

class ParameterGetter{
    NeuralNetwork& nn;

    std::string file_weights1;
    std::string file_weights2;

    std::string file_biases1;
    std::string file_biases2;

    public:
    ParameterGetter(NeuralNetwork& nn, std::string w1_file,std::string w2_file, std::string b1_file, std::string b2_file):nn(nn), file_weights1(w1_file),  file_weights2(w2_file),  file_biases1(b1_file),  file_biases2(b2_file){}

    void putParams();

    void getParams();
};

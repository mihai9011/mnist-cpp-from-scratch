    #include "nn.hpp"
    #include <iostream>
    #include <fstream>
    #include <vector>

    //?DEBUG: A0 size: 784x60000
    //?DEBUG: W0 size: 16x784
    //?DEBUG: b0 size: 16x1
    //?DEBUG: Z1 calculat size: 16x60000



    void NeuralNetwork::add_layer(int ant_neurons , int curr_neurons){
        Layer layer(ant_neurons, curr_neurons);
        layers.push_back(layer);
    }

    Eigen::MatrixXd NeuralNetwork::ReLU(Eigen::MatrixXd Z1){
        return Z1.unaryExpr([] (double x){
            return (x > 0)? x : 0; 
        });
    }

    Eigen::MatrixXd NeuralNetwork::ReLU_derivative(Eigen::MatrixXd Z){
        //if the activation is bigger than 0 => ReLU(x) = x => ReLU`(x) = 1
        //else ReLU(x) = 0 => ReLU`(x) = 0

        return Z.unaryExpr([] (double x){
            return  static_cast<double>(x > 0); 
        });
    }

    Eigen::MatrixXd NeuralNetwork::softmax(Eigen::MatrixXd Z2){
        Eigen::MatrixXd stabilized = Z2.rowwise() - Z2.colwise().maxCoeff();
        
        // 2. Exponențiere element cu element
        Eigen::MatrixXd exp_mat = stabilized.array().exp();
        
        // 3. Împărțire la suma fiecărei coloane
        Eigen::MatrixXd softmax_col = exp_mat.array().rowwise() / exp_mat.colwise().sum().array();

        return softmax_col;
    }

    void NeuralNetwork::forwardprop(){
        
        //1.calcul Z, sume ponderate layer 1

        Eigen::MatrixXd Z1 = layers[0].W * A0;


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

    void NeuralNetwork::show_predicted_numb()const{

        Eigen::Index maxRow;
        layers[1].A.col(0).maxCoeff(&maxRow);  

        if(layers[1].A(maxRow, 0) > 0.6){ 
            std::cout<<maxRow<<'\n';
        }
        else{

            double max1 = layers[1].A(0,0);
            int numb1 = 0;

            double max2 = layers[1].A(1,0);
            int numb2 = 1;

            double max3 = layers[1].A(2,0);
            int numb3 = 2;

            for(Eigen::Index i =3; i < layers[1].A.col(0).size(); ++i){
                double val = layers[1].A(i, 0);

                if(val > max1 && val > max2 && val > max3){
                    max3 = max2;
                    numb3 =numb2;

                    max2 = max1;
                    numb2 = numb1;

                    max1 = val;
                    numb1 = i;
                }
                else if(val > max2 && val > max3){
                    max3 = max2;
                    numb3 = numb2;

                    max2 = val;
                    numb2 = i;
                }
                else if(val > max3){
                    max3 = val;
                    numb3 = i;
                }
            } 
            std::cout<<numb1 << ": " << max1 <<"||" <<numb2 << ": " << max2 <<"||" <<numb3 << ": " << max3 <<"||"<<'\n';  
        }
    }   

    double NeuralNetwork::getLearningRate() const{
        return learning_rate;
    }

    void NeuralNetwork::run(){
        forwardprop();
    }

    Eigen::MatrixXd NeuralNetwork::getA0(){
        return A0;
    }


    //Parameter Getter


    void ParameterGetter::putParams(){
        std::ofstream w1(file_weights1);
        std::ofstream w2(file_weights2);

        std::ofstream B1(file_biases1);
        std::ofstream B2(file_biases2);


        w1 << nn.layers[0].W;
        w2 << nn.layers[1].W;
        
        B1 << nn.layers[0].b;
        B2 << nn.layers[1].b;
        
    }


    void ParameterGetter::getParams(){
        std::ifstream w1(file_weights1);
        std::ifstream w2(file_weights2);

        std::ifstream B1(file_biases1);
        std::ifstream B2(file_biases2);

        //updating biases
        for(int i = 0; i < 16; i++){
            B1 >> nn.layers[0].b(i);
        }

        for(int i =0; i < 10; i++){
            B2 >> nn.layers[1].b(i);
        }

        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 784; ++j) {
                w1 >> nn.layers[0].W(i, j);
            }
        }

        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 16; ++j) {
                w2 >> nn.layers[1].W(i, j);
            }
        }

    }


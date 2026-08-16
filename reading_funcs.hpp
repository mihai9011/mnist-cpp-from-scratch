#pragma once
#include <Eigen/Dense>

int reverse_int(int i);
void read_input(Eigen::MatrixXd& A0);
void fetch_data_training_set(Eigen::MatrixXd& A0, Eigen::RowVectorXd& Y);
void fetch_data_10k(Eigen::MatrixXd& A0, Eigen::RowVectorXd& Y);


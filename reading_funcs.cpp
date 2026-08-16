#include <Eigen/Dense>
#include <fstream>
#include <iostream>

//pt reteaua normala
void read_input(Eigen::MatrixXd& A0){
    std::ifstream input("input.txt");

    A0.resize(784, 1);

    for(int i = 0; i < 784; i++){
        input >> A0(i, 0);
    }

}



//pt cea trainable
int reverse_int(int i) {
    unsigned char c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void fetch_data_training_set(Eigen::MatrixXd& A0, Eigen::RowVectorXd& Y){

    // A0   
    std::ifstream img_file("data/train-images.idx3-ubyte", std::ios::binary);
    if (!img_file.is_open()) {
        std::cout << "Opening training data file error!\n";
        return;
    }

    int magic = 0, n_images = 0, rows = 0, cols = 0;
    img_file.read(reinterpret_cast<char*>(&magic), 4);
    img_file.read(reinterpret_cast<char*>(&n_images), 4);
    img_file.read(reinterpret_cast<char*>(&rows), 4);
    img_file.read(reinterpret_cast<char*>(&cols), 4);

    n_images = reverse_int(n_images);
    rows = reverse_int(rows);
    cols = reverse_int(cols);

    int image_size = rows * cols; 

    A0.resize(image_size, n_images);

    for (int i = 0; i < n_images; ++i) {
        for (int p = 0; p < image_size; ++p) {
            unsigned char pixel = 0;
            img_file.read(reinterpret_cast<char*>(&pixel), 1);
            A0(p, i) = static_cast<double>(pixel) / 255.0;
        }
    }
    img_file.close();

    // Y
    std::ifstream lbl_file("data/train-labels.idx1-ubyte", std::ios::binary);
    if (!lbl_file.is_open()) {
        std::cout << "Opening training labels error!\n";
        return;
    }

    lbl_file.read(reinterpret_cast<char*>(&magic), 4);
    lbl_file.read(reinterpret_cast<char*>(&n_images), 4);
    n_images = reverse_int(n_images);

    Y.resize(n_images);

    for (int i = 0; i < n_images; ++i) {
        unsigned char label = 0;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        Y(0, i) = static_cast<double>(label);
    }
    lbl_file.close();
}

void fetch_data_10k(Eigen::MatrixXd& A0, Eigen::RowVectorXd& Y){
    // A0
    std::ifstream img_file("data/t10k-images.idx3-ubyte", std::ios::binary);
    if (!img_file.is_open()) {
        std::cout << "Opening testing data file error!\n";
        return;
    }

    int magic = 0, n_images = 0, rows = 0, cols = 0;

    img_file.read(reinterpret_cast<char*>(&magic), 4);
    img_file.read(reinterpret_cast<char*>(&n_images), 4);
    img_file.read(reinterpret_cast<char*>(&rows), 4);
    img_file.read(reinterpret_cast<char*>(&cols), 4);

    n_images = reverse_int(n_images);
    rows = reverse_int(rows);
    cols = reverse_int(cols);

    int image_size = rows * cols; // 784

    A0.resize(image_size, n_images);

    for (int i = 0; i < n_images; ++i) {
        for (int p = 0; p < image_size; ++p) {
            unsigned char pixel = 0;
            img_file.read(reinterpret_cast<char*>(&pixel), 1);
            A0(p, i) = static_cast<double>(pixel) / 255.0;
        }
    }
    img_file.close();

    // Y
    std::ifstream lbl_file("data/t10k-labels.idx1-ubyte", std::ios::binary);
    if (!lbl_file.is_open()) {
        std::cout << "Opening testing labels error!\n";
        return;
    }

    lbl_file.read(reinterpret_cast<char*>(&magic), 4);
    lbl_file.read(reinterpret_cast<char*>(&n_images), 4);
    n_images = reverse_int(n_images);

    Y.resize(n_images);

    for (int i = 0; i < n_images; ++i) {
        unsigned char label = 0;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        Y(0, i) = static_cast<double>(label);
    }
    lbl_file.close();

}


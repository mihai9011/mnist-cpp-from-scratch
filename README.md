MLP network designed for recognizing digits


DATASET

- MNIST digits dataset.
- using the predefined split for training/testing data.
- the pixel values were normalized by Min-Max Scaling with the minimum value 0 and maximum value 255 (division of the pixel value by 255).
- labels where One-Hot encoded.


THE NEURAL NETWORKS ARCHITECTURE

- this neural network is a simple, yet highly accurate multilayer perceptron network implemented from scratch in C++.
- it is comprised of a input layer of 784 neurons, a hidden layer of 16 neurons and the ouput layer containing 10 neurons with probabilities for each digit.
- if the biggest probability of the output layer is less or equal to 0.6, the layer will show its top 3 guesses


PYTHON SCRIPT

- using pygame, i made a graphical user iterface, so the network can be tested.
- it centers  the user's digit, since the teh model was pre-trained on centered digits
- the script saves the normalized and centered data, created by the user's digit, in input.txt, and then calls for the nn.exe executable, which will make the prediction.


HOW TO TEST IT

- create a virtual environmnent, using py

- activate it

- create the executable, running this command in the terminal:

            g++ main.cpp nn.cpp reading_funcs.cpp tnn.cpp -I . -o nn.exe

(it must be named nn.exe so the py script can find it)

- run the python script with the command:

            python gui.py

- a 280x280 window should appear:

    SPACE - to guess
    C - to clear the window


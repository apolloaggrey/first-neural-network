# Neural Network in C++
A neural network implemented with **matrices** in C++, from scratch !

This program is meant to be used for supervised learning.


# Network Class
The Network class contains the gradient descent algorithm.

    // make prediction
    Matrix<double> computeOutput(std::vector<double> input);

    // learns from the previous computeOutput()
    void learn(std::vector<double> expectedOutput);

    // save all network's parameters into a file (after a training)
    void saveNetworkParams(const char *filepath);

    // load network's parameters from a file so you don't have to train it again
    void loadNetworkParams(const char *filepath);
    // or use the constructor
    Network(const char *filepath);




Amen !

#ifndef DEF_NETWORK
#define DEF_NETWORK

#include <vector>
#include <fstream>
#include <cmath>
#include <time.h>
#include <stdlib.h>

#include "../Matrix/Matrix.h"

class Network
{
public:
    Network(int inputNeuron, int hiddenNeuron, int outputNeuron, double learningRate);
    Network(const char *filepath);
    Matrix<double> computeOutput(std::vector<double> input);
    void learn(std::vector<double> expectedOutput);

    void saveNetworkParams(const char *filepath);
    void loadNetworkParams(const char *filepath);

private:
    Matrix<double> X;
    Matrix<double> W1;
    Matrix<double> H;
    Matrix<double> W2;
    Matrix<double> Y;
    Matrix<double> Y2;
    Matrix<double> E1;
    Matrix<double> E2;
    Matrix<double> dEdW1;
    Matrix<double> dEdW2;
    Matrix<double> T1;
    Matrix<double> T2;

    double learningRate;

    static double random(double x);
    static double sigmoid(double x);
    static double sigmoidePrime(double x);
};

#endif

//
// Created by AGGREY APOLLO on 12/29/2019.
//
#include <iostream>
#include <sstream>
#include "Matrix.cpp"
#include <random>
#include <cmath>
#include <utility>

class Network
        {
private:
    Matrix X,Y2;
    int input;
    std::vector<std::vector<double> > inputVector;
    std::vector<std::vector<double> > outputVector;
    std::vector<int> hidden_layers;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    std::vector<Matrix> dJdB;
    std::vector<Matrix> dJdW;
    std::vector<Matrix> HY;
    double  learningRate;
    int training_Size;
    int output;
    unsigned int epoch = 30;


public:




    Network(int input,std::vector<int>&hidden_layers,int output, double learning_rate = 0.1) // 16,[15,10,5],2
    {
        this->input=input;
        this->output=output;
        this->hidden_layers=hidden_layers;
        this->learningRate = learning_rate;
        this->weights = std::vector<Matrix> (hidden_layers.size()+1);
        this->biases = std::vector<Matrix> (hidden_layers.size()+1);
        this->dJdB = std::vector<Matrix> (hidden_layers.size()+1);
        this->dJdW = std::vector<Matrix> (hidden_layers.size()+1);
        this->HY = std::vector<Matrix> (hidden_layers.size()+1);
        //TODO: Automatically generate X, Y, H, W1, W2, B1, B2, Y2, dJdB1, dJdB2, dJdW1, dJdW2;; in one VECTOR
        std::cout<<"creating Network...\n";
    }

    void load_Data(const char *filename, int Data_size)
    {
        this->training_Size = Data_size;
        inputVector.resize(training_Size);
        outputVector.resize(training_Size);
        std::ifstream file(filename);
        if(file.is_open())
        {
            std::cout<<"loading data...\n";
            std::string line;
            float n;

            for (int i = 0; i < training_Size; i++) // load 10 examples
            {
                getline(file, line);
                n = std::stof(line.substr(0, 1)); // get the number that is represented by the array
//                std::cout<<n<<"\n";
                outputVector[i].resize(10); // output is a vector of size 10 because there is 10 categories (0 to 9)
                outputVector[i][n] = 1; // set value to one at the index that represents the number. Other values are automatically 0 because of the resize()

                for (int h = 0; h < 28; h++) // 'images' are 28*28 pixels
                {
                    for (int w = 0; w < 28; w++)
                    {
//                    std::cout<<line.substr(28*h+w+1, 1).c_str();
                        inputVector[i].push_back(std::stof(line.substr(28*h+w+1, 1)));
                    }
//                std::cout<<"\n";
                }
            }
        }
        file.close();
//        std::cout<<"loading data complete...\n";
    }

    static double random(double x)
    {
        return (double)(rand() % 10000 + 1)/10000-0.5; // NOLINT(cert-msc30-c)
    }

    void init()
    {
        std::cout<<"creating W"<<0<<"\n";
        weights[0] = Matrix(input, hidden_layers[0]);
        int x =0;
        for(; x < hidden_layers.size()-1; x++)
        {
            std::cout<<"creating W"<<x+1<<"\n";
            weights[x+1] = Matrix(hidden_layers[x], hidden_layers[x+1]);
        }
        std::cout<<"creating W"<<x+1<<"\n";
        weights[x+1] = Matrix(hidden_layers[x], output);

        x=0;
        for(; x < hidden_layers.size(); x++)
        {
            std::cout<<"creating B"<<x<<"\n";
            biases[x] = Matrix(1, hidden_layers[x]);
        }
        std::cout<<"creating B"<<x<<"\n";
        biases[x] = Matrix(1, output);

        std::cout<<"randomizing Weights\n";
        for(auto & weight : weights)
        {
            weight = weight.applyFunction(random);
        }

        std::cout<<"randomizing Biases...\n";
        for(auto & bias : biases)
        {
            bias = bias.applyFunction(random);
        }
    }


    static double sigmoid(double x)
    {
        return 1/(1+exp(-x));
    }

    Matrix computeOutput(std::vector<double> in)
    {
        X = Matrix({std::move(in)});
//      X = Matrix({input}); // row matrix
        HY[0] = X.dot(weights[0]).add(biases[0]).applyFunction(sigmoid);
//      H = X.dot(W1).add(B1).applyFunction(sigmoid);

        for(int x = 1;x<biases.size();x++)
        {
            HY[x] = HY[x-1].dot(weights[x]).add(biases[x]).applyFunction(sigmoid);
        }
        return HY[HY.size()-1];

    }


    std::vector<double> computeInput(Matrix output)
    {


    }





    static double sigmoidePrime(double x)
    {
        return exp(-x)/(pow(1+exp(-x), 2));
    }

    void learn(std::vector<double> expectedOutput)
    {
//        std::cout<<"learning...\n";
        Y2 = Matrix({std::move(expectedOutput)}); // row matrix
        // We need to calculate the partial derivative of J, the error, with respect to W1,W2,B1,B2

        // compute gradients
//        std::cout<<"compute gradients...\n";

        dJdB[biases.size()-1] = HY[HY.size()-1].subtract(Y2).multiply(HY[HY.size()-2].dot(weights[weights.size()-1]).add(biases[biases.size()-1]).applyFunction(sigmoidePrime));
        int y=3;
        for(unsigned int x =(dJdB.size()-2);x>0;x--)
        {
            dJdB[x] = dJdB[x+1].dot(weights[x+1].transpose()).multiply(HY[HY.size()-y].dot(weights[x]).add(biases[x]).applyFunction(sigmoidePrime));
            y++;
        }
        dJdB[0] = dJdB[1].dot(weights[1].transpose()).multiply(X.dot(weights[0]).add(biases[0]).applyFunction(sigmoidePrime));



        y=2;
        for(unsigned int x =(dJdB.size()-1);x>0;x--)
        {
            dJdW[x] = HY[HY.size()-y].transpose().dot(dJdB[x]);
            y++;
        }
        dJdW[0] = X.transpose().dot(dJdB[0]);



        // update weights
//        std::cout<<"update weights...";
        for(int x = 0;x<weights.size();x++)
        {
            weights[x] = weights[x].subtract(dJdW[x].multiply(learningRate));
        }

        for(int x = 0;x<biases.size();x++)
        {
            biases[x] = biases[x].subtract(dJdB[x].multiply(learningRate));
        }
    }

    static double stepFunction(double x)
    {
        if(x>0.8)
            return 1.0;
        if(x<0.2)
            return 0.0;
        else
            return x;
    }

    static std::string format(double x)
    {
        std::string s = " ";
        s += std::to_string(x)+ " ";
        return s;
    }

    void test(float percent)
    {
        // test
        std::cout << std::endl << "expected output : actual output "<< std::endl;
        for (unsigned long long i=(training_Size-int(0.01*percent*training_Size)); i<inputVector.size() ; i++) // testing on last 10 examples
        {
            // as the sigmoid function never reaches 0.0 nor 1.0
            // it can be a good idea to consider values greater than 0.9 as 1.0 and values smaller than 0.1 as 0.0
            // hence the step function.
            for (int j=0 ; j<10 ; j++)
            {
                std::cout << outputVector[i][j] << "";
            }
            Matrix result = computeOutput(inputVector[i]).applyFunction(stepFunction);
            std::cout << ": " << result<<std::endl;
        }
    }

    void train(const std::string& dir,unsigned int epochs=30,int percent=90)
    {
        this->epoch=epochs;
        for (int i=0 ; i<epoch ; i++)
        {
            std::cout << "Epoch #" << i+1 <<"/" <<epoch<< std::endl;
            std::cout << "training:" ;
            int x = int(0.01*percent*training_Size);
            int y = x/10;
            for (unsigned long long j=0 ; j<x; j++) // skip the last 10 examples to test the program at the end
            {
                computeOutput(inputVector[j]);
                learn(outputVector[j]);
                if(j%y==0)std::cout << ".";
            }
            std::cout << "\n";
//            TODO: TEST AND DISPLAY ACCURACY AT EVERY EPOCH
//            std::cout << "Saving Epoch #" << i+1 <<" Model..." << std::endl;
            /// save model
            for (int k = 0; k < weights.size(); k++)
            {
                std::stringstream ss;
                ss << dir<<R"(\W)"<<k<<".txt";
                std::string file = ss.str();
                weights[k].serialize(file);
            }
            for (int k = 0; k < biases.size(); k++)
            {
                std::stringstream ss;
                ss << dir<<R"(\B)"<<k<<".txt";
                std::string file = ss.str();
                biases[k].serialize(file);
            }
        }
    }

    void load_model(const std::string& dir)
    {
        std::cout << "Loading Weights...";
        for (int k = 0; k < weights.size(); k++)
        {
            std::stringstream ss;
            ss << dir<<R"(\W)"<<k<<".txt";
            std::string file = ss.str();
            weights[k]=file;
        }
        std::cout << "Loading Biases...";
        for (int k = 0; k < biases.size(); k++)
        {
            std::stringstream ss;
            ss <<dir<<R"(\B)"<<k<<".txt";
            std::string file = ss.str();
            biases[k]=file;
        }
    }

};

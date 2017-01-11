# Neural Network in C++
A neural network implemented with **matrices** in C++, without any third party library !

# What's in there ?

+ **src/XOR :** Learning XOR operation.
+ **src/Plot :** XOR version but prints the weights/error variations in files to plot them later (see images below).
+ **src/Digits-Recognition :** Learning to recognize hand-written digits with a training file.

# Download, Compile & Run
    git clone https://github.com/omaflak/Neural-Network
    cd Neural-Network/src
    git clone https://github.com/omaflak/Matrix

    # cd into one of the directories above and:
    sh compile.sh
    ./main

# Network Class
The Network class contains the gradient descent algorithm.

Both **src/XOR** and **src/Digit-Recognition** are using it. Here is the header file:

    // constructor
    Network(int inputNeuron, int hiddenNeuron, int outputNeuron, double learningRate);

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



# Plot

This version is not using the Network class, it is a one file program and it was only to plot the network's parameters on a graph.

The program is learning XOR operation and is saving some weights and the error over time in files so we can plot them later.

Once the program has finished, 4 files should be created in the current directory: plotX, plotY, plotEX, plotEY

+ **plotX/Y :** xy coordinates for weights variation

+ **plotEX/EY :** xy coordinates for error variation

You can plot the data with **[plotly](https://plot.ly/create/)** by pressing the **import** button (top right).

**I had some trouble with Plotly on Google Chrome. Switching to another browser fixed the problem...**

Here is some plot :

![alt tag](https://github.com/omaflak/Neural-Network/blob/master/images/weightsPlot.png?raw=true)
![alt tag](https://github.com/omaflak/Neural-Network/blob/master/images/errorPlot.png?raw=true)

We can see that the program is actually working: while the weights are converging to specific values, the error is decreasing.

Amen !

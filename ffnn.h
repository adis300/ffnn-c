/*
 * ffnn.h
 *
 * Copyright (c) 2018 Disi A
 * 
 * Author: Disi A
 * Email: adis@live.cn
 *  https://www.mathworks.com/matlabcentral/profile/authors/3734620-disi-a
 */

#ifndef _ffnn_h
#define _ffnn_h


#ifdef __cplusplus
extern "C" {
#endif

typedef double (*ActivationFunc)(double z);

typedef struct {

    ActivationFunc activation_func;

    int number_of_nodes;

    int input_length;

    double *weights; // (numberOfNodes * inputLength) A grid, row1, row2, row3 ... to represent a matrix

    double *biases; // (NumberOfNodes) A Vector, bias1, bias2, bias3 ... to represent a vector

    double *output; // (NumberOfNodes) A Vector, bias1, bias2, bias3 ... to represent a vector

} NetworkLayer;

typedef struct {
    /* Total number of weights, and size of weights buffer. */
    int output_length;

    /* Total number of neurons + inputs and size of output buffer. */
    int input_length;

    /* All weights (total_weights long). */
    double * output;

    /* Stores input array and output of each neuron (total_neurons long). */
    int layer_length;

    /* Stores delta of each hidden and output neuron (total_neurons - inputs long). */
    NetworkLayer * layers;

} Network;

double ffnn_activation_sigmoid(double x);
double ffnn_activation_threshold(double x);
double ffnn_activation_linear(double x);
double ffnn_activation_relu(double x);

// Layer functions
NetworkLayer* create_layer(int numberOfNodes, int inputLength, double* weights, double * biases, const char* activation);
void free_layer(NetworkLayer* network_layer);
double * run_layer(NetworkLayer* network_layer, double* input);

double * run (Network* network, double * input);

#ifdef __cplusplus
}
#endif

#endif /* _ffnn_h */

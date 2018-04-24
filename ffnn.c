
/*
 * ffnn.c
 *
 * Copyright (c) 2018 Disi A
 * 
 * Author: Disi A
 * Email: adis@live.cn
 *  https://www.mathworks.com/matlabcentral/profile/authors/3734620-disi-a
 */

#include "ffnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "extra/jsmn.c"

#define LOOKUP_SIZE 4096
#define SIGMOID_CUTOFF 45.0

// const double SIGMOID_DOM_MIN = -20.0;
// const double SIGMOID_DOM_MAX = 20.0;

double sigmoid_lookup[LOOKUP_SIZE];

// Activation functions
double inline ffnn_activation_linear(double x) { return x;}

double inline ffnn_activation_threshold(double x) {return x > 0;}

double inline ffnn_activation_relu(double x) {
    if(x > 0) return x;
    return 0;
}

double inline ffnn_activation_sigmoid(double x) {
    if (x < -SIGMOID_CUTOFF) return 0;
    if (x > SIGMOID_CUTOFF) return 1;
    return 1.0 / (1 + exp(-x));
}

NetworkLayer* create_layer(int number_of_nodes, int input_length, double* weights, double * biases, const char* activation){
    // Layer activation function defaults to sigmoid
    NetworkLayer* network_layer = (NetworkLayer *) malloc(sizeof(NetworkLayer));
    network_layer -> number_of_nodes = number_of_nodes;
    network_layer -> input_length = input_length;

    // Initialize activation function
    if (strcmp(activation, "linear") == 0) network_layer -> activation_func = ffnn_activation_linear;
    else if (strcmp(activation, "relu") == 0) network_layer -> activation_func = ffnn_activation_relu;
    else if (strcmp(activation, "threshold") == 0) {
        printf("NetworkLayer:create_layer:loading a threshold function\n");
        network_layer -> activation_func = ffnn_activation_threshold;
    }
    else {
        printf("NetworkLayer:create_layer:loading a sigmoid function as default\n");
        network_layer -> activation_func = ffnn_activation_sigmoid;
    }
    // Initialize weights, biases and output
    network_layer -> weights = weights; //(double*) realloc(weights, number_of_nodes * input_length * sizeof(double));
    network_layer -> biases = biases; //(double*) realloc(biases, number_of_nodes * sizeof(double));
    network_layer -> output = (double*) malloc(number_of_nodes * sizeof(double));
    return network_layer;
}

void free_layer(NetworkLayer* network_layer){
    free(network_layer -> weights);
    free(network_layer -> biases);
    free(network_layer -> output);
    free(network_layer);
}

double * run_layer(NetworkLayer* network_layer, double* input){
    double res;
    for(int node = 0; node < network_layer -> number_of_nodes; node ++){
        res = network_layer -> biases[node];
        for(int i = 0; i < network_layer -> input_length; i ++){
            res += network_layer -> weights[node * network_layer -> input_length + i] * input[i];
        }
        network_layer -> output[node] = network_layer -> activation_func(res);
    }
    return network_layer -> output;
}

/*
void create_ffnn_sigmoid_lookup() {
    const double f = (SIGMOID_DOM_MAX - SIGMOID_DOM_MIN) / LOOKUP_SIZE;
    double interval = LOOKUP_SIZE / (SIGMOID_DOM_MAX - SIGMOID_DOM_MIN);
    for (int i = 0; i < LOOKUP_SIZE; ++i) {
        sigmoid_lookup[i] = ffnn_activation_sigmoid(SIGMOID_DOM_MIN + f * i);
    }
}
*/

Network* create_network(char * json_network){
    Network* network = (Network *) malloc(sizeof(Network));
    printf("Network:create_network:\n %s \n\n", json_network);

    return network;
}

double * run (Network* network, double * input){
    return 0;
}


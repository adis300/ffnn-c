
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
#include <math.h>
#include <string.h>

#define LOOKUP_SIZE 4096
#define SIGMOID_CUTOFF 45.0

const double SIGMOID_DOM_MIN = -20.0;
const double SIGMOID_DOM_MAX = 20.0;

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

NetworkLayer* create_layer(int numberOfNodes, int inputLength, double* weights, double * biases, const char* activation){
    
    return 0;
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

double * layer_output (int col, int row, double * input, double * grid, double * bias) {

}


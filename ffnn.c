
/*
 * ffnn.c
 *
 * Copyright (c) 2018 Disi A
 * 
 * Author: Disi A
 * Email: adis@live.cn
 *  https://www.mathworks.com/matlabcentral/profile/authors/3734620-disi-a
 */

#include <math.h>

// Activation functions
double inline ffnn_activation_linear(double x) { return x;}

double inline ffnn_activation_threshold(double x) {return x > 0;}

double inline ffnn_activation_relu(double x) {
    if(x > 0) return x;
    return 0;
}

double inline ffnn_activation_sigmoid(double x) {
    if (x < -45.0) return 0;
    if (x > 45.0) return 1;
    return 1.0 / (1 + exp(-x));
}


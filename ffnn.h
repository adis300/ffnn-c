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

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

double ffnn_activation_sigmoid(double x);
double ffnn_activation_threshold(double x);
double ffnn_activation_linear(double x);
double ffnn_activation_relu(double x);

#ifdef __cplusplus
}
#endif

#endif /* _ffnn_h */

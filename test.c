#include "ffnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int numberOfNodes = 3;
const int inputLength = 4;

static char * JSON_NETWORK = 
    //"{\"layerSizes\":[3,2],\"activation\":\"sigmoid\",\"activations\":[\"sigmoid\",\"relu\"],\"weights\":[{\"col\": 3,\"row\": 2,\"grid\":[0.016354116618798153,-0.35975899610657203, -0.4739052054816415, 0.90403169668467331, 0.3511014503199148, -0.73113043081533657]}],\"biases\":[{\"vector\" :[ 0.27921540303119241, 0.37599578750133311]},{\"vector\" :[ 0.27921540303119241, 0.37599578750133311]}]}";
    "{\"layerSizes\":[3,2],\"activation\":\"sigmoid\",\"activations\":[\"sigmoid\",\"relu\"],\"biases\":[{\"vector\" :[ 0.27921540303119241, 0.37599578750133311]}],\"weights\":[{\"col\": 3,\"row\": 2,\"grid\":[0.016354116618798153,-0.35975899610657203, -0.4739052054816415, 0.90403169668467331, 0.3511014503199148, -0.73113043081533657]}]}";

int test_layer(){

    double * biases = (double *) malloc(numberOfNodes * sizeof(double));
    biases[0] = 1.0; biases[1] = 2.0; biases[2] = 3.0;

    double * inputs = (double *) malloc(inputLength * sizeof(double));
    inputs[0] = 1.0; inputs[1] = 2.0; inputs[2] = 3.0; inputs[3] = 4.0;

    double * weights = (double *) malloc(numberOfNodes * inputLength * sizeof(double));
    weights[0] = 1.0; weights[1] = 2.0; weights[2] = 3.0; weights[3] = 4.0;
    weights[4] = 1.0; weights[5] = 2.0; weights[6] = 3.0; weights[7] = 4.0;
    weights[8] = 1.0; weights[9] = 2.0; weights[10] = 3.0; weights[11] = 4.0;

    NetworkLayer* layer = create_layer(numberOfNodes, inputLength, weights, biases, "relu");
    // Test activation functions
    double activated = layer -> activation_func(0.1);
    if(fabs(activated - 0.1) > 0.001) {
        printf("ERROR:test_create_layer:positive activation:%lf \n", activated);
        return 1;
    }
    activated = layer -> activation_func(-0.1);
    if(fabs(activated - 0) > 0.001) {
        printf("ERROR:test_create_layer:negative activation:%lf \n", activated);
        return 1;
    }

    double * output = run_layer(layer, inputs);
    if(fabs(output[0] - 31.0) > 0.001 || fabs(output[1] - 32.0) > 0.001 || fabs(output[2] - 33.0) > 0.001){
        printf("ERROR:test_create_layer:incorrect layer response:%lf,%lf, %lf \n", output[0],output[1],output[2]);
    }

    free(inputs);
    free_layer(layer);
    return 0;
}

int test_create_network(){
    Network* network = create_network(JSON_NETWORK);
    if(network == NULL) return 1;
    return 0;
}

int main () //(int argc, char *argv[])
{   
    printf("========= Starting ffnn test =========\n");

    if(test_layer() != 0) {
        printf("FAILURE:test_layer.\n");
        return 1;
    }

    if(test_create_network() != 0) {
        printf("FAILURE:test_create_network.\n");
        return 1;
    }

    printf("SUCCESS:No failure detected.\n");
    return 0;
}


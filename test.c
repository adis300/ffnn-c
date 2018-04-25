#include "ffnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int numberOfNodes = 3;
const int inputLength = 4;

static char * JSON_NETWORK = 
    //"{\"layerSizes\":[3,2],\"activation\":\"sigmoid\",\"activations\":[\"relu\"],\"weights\":[{\"col\": 3,\"row\": 2,\"grid\":[0.016354116618798153,-0.35975899610657203, -0.4739052054816415, 0.90403169668467331, 0.3511014503199148, -0.73113043081533657]}],\"biases\":[{\"vector\" :[ 0.27921540303119241, 0.37599578750133311]}]}";
    "{\"layerSizes\":[3,2,2],\"activations\":[\"relu\",\"relu\"],\"weights\":[{\"col\": 3,\"row\": 2,\"grid\":[1,1, 1.0, 1.0, 1.0, 1.0]},{\"col\": 2,\"row\": 2,\"grid\":[1,1,1,1]}],\"biases\":[{\"vector\" :[ 0.25, 0.25]},{\"vector\" :[ 0.5, 0.5]}]}";

    //"{\"layerSizes\":[3,2],\"activation\":\"sigmoid\",\"activations\":[\"sigmoid\",\"relu\"],\"weights\":[{\"col\": 3,\"row\": 2,\"grid\":[0.016354116618798153,-0.35975899610657203, -0.4739052054816415, 0.90403169668467331, 0.3511014503199148, -0.73113043081533657]},{\"col\": 3,\"row\": 2,\"grid\":[0.016354116618798153,-0.35975899610657203, -0.4739052054816415, 0.90403169668467331, 0.3511014503199148, -0.73113043081533657]}],\"biases\":[{\"vector\" :[ 0.27921540303119241, 0.37599578750133311]}]}";

int test_layer(){

    double * biases = (double *) malloc(numberOfNodes * sizeof(double));
    biases[0] = 1.0; biases[1] = 2.0; biases[2] = 3.0;

    double input[4] = {1.0, 2.0, 3.0, 4.0};

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

    double * output = run_layer(layer, (double*) &input);
    if(fabs(output[0] - 31.0) > 0.001 || fabs(output[1] - 32.0) > 0.001 || fabs(output[2] - 33.0) > 0.001){
        printf("ERROR:test_create_layer:incorrect layer response:%lf,%lf, %lf \n", output[0],output[1],output[2]);
        return 1;
    }

    // free(inputs);
    free_layer(layer);
    return 0;
}

int test_create_network(){
    Network* network = create_network(JSON_NETWORK);
    if(network == NULL) return 1;

    double input[3] = {1.0, 1.0, 1.0};

    double * output = run_network(network, (double*) &input);

    if(fabs(output[0] - 7.0) > 0.001 || fabs(output[1] - 7.0) > 0.001){
        printf("ERROR:test_create_network:incorrect network response:%lf,%lf \n", output[0],output[1]);
        return 1;
    }
    return 0;
}

int main () //(int argc, char *argv[])
{   
    printf("========= Starting ffnn test =========\n");

    if(test_layer() == 0) {
        printf("*** SUCCESS:test_layer.\n");
    }else{
        printf("*** FAILURE:test_layer!\n");
        return 1;
    }

    if(test_create_network() == 0) {
        printf("*** SUCCESS:test_create_network.\n");
    }else{
        printf("*** FAILURE:test_create_network!\n");
        return 1;
    }

    printf("========= SUCCESS:No failure detected. =========\n");
    return 0;
}


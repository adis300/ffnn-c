#include "ffnn.h"
#include <stdlib.h>
#include <stdio.h>

static char * JSON_NETWORK_ACTIVATION_BY_LAYERS = "{\"layerSizes\":[3,2,2],\"activations\":[\"relu\",\"relu\"],\"weights\":[{\"col\": 3,\"row\": 2,\"grid\":[1,1, 1.0, 1.0, 1.0, 1.0]},{\"col\": 2,\"row\": 2,\"grid\":[1,1,1,1]}],\"biases\":[{\"vector\" :[ 0.25, 0.25]},{\"vector\" :[ 0.5, 0.5]}]}";

static char * JSON_NETWORK_UNIVERSAL_ACTIVATION = "{\"layerSizes\":[3,2,2],\"activation\":\"sigmoid\",\"weights\":[{\"col\": 3,\"row\": 2,\"grid\":[1,1, 1.0, 1.0, 1.0, 1.0]},{\"col\": 2,\"row\": 2,\"grid\":[1,1,1,1]}],\"biases\":[{\"vector\" :[ 0.25, 0.25]},{\"vector\" :[ 0.5, 0.5]}]}";

void activation_by_layer(){

    printf("----- EXAMPLE1: Activation by layer example -----\n");

    Network* network = create_network(JSON_NETWORK_ACTIVATION_BY_LAYERS);

    double input[3] = {1.0, 1.0, 1.0};
    double * output = run_network(network, (double*) &input);

    printf("Network response:%lf,%lf \n", output[0],output[1]);

    printf("Network saved response:%lf,%lf \n", network -> output[0], network -> output[1]);
    printf("----- Activation by layer done. -----\n\n");
}

void activation_universal(){

    printf("----- EXAMPLE2:Activation universal example -----\n");

    Network* network = create_network(JSON_NETWORK_UNIVERSAL_ACTIVATION);

    // double * input = (double *) alloca(3 * sizeof(double));
    double input[3] = {1.0, 1.0, 1.0};
    double * output = run_network(network, (double*) &input);

    printf("Network response:%lf,%lf \n", output[0],output[1]);

    printf("Network saved response:%lf,%lf \n", network -> output[0], network -> output[1]);
    printf("----- Activation universal done. -----\n\n");
}

int main() {   
    printf("========= FFNN example =========\n\n");
    activation_by_layer();
    activation_universal();
    printf("========= Done. =========\n\n");
    return 0;
}


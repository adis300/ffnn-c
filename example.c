#include "ffnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static char * JSON_NETWORK = "{\"layerSizes\":[3,2,2],\"activations\":[\"relu\",\"relu\"],\"weights\":[{\"col\": 3,\"row\": 2,\"grid\":[1,1, 1.0, 1.0, 1.0, 1.0]},{\"col\": 2,\"row\": 2,\"grid\":[1,1,1,1]}],\"biases\":[{\"vector\" :[ 0.25, 0.25]},{\"vector\" :[ 0.5, 0.5]}]}";

int main() {   
    printf("========= FFNN example =========\n");

    Network* network = create_network(JSON_NETWORK);

    double * input = (double *) alloca(3 * sizeof(double));
    input[0] = 1.0; input[1] = 1.0; input[2] = 1.0;
    double * output = run_network(network, input);

    printf("Network response:%lf,%lf \n", output[0],output[1]);

    printf("Network saved response:%lf,%lf \n", network -> output[0], network -> output[1]);

    printf("========= Done. =========\n");
    return 0;
}


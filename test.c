#include "ffnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "extra/network.pb-c.h"

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
    Network* network = create_network_from_json(JSON_NETWORK);
    if(network == NULL) return 1;

    double input[3] = {1.0, 1.0, 1.0};

    double * output = run_network(network, (double*) &input);

    if(fabs(output[0] - 7.0) > 0.001 || fabs(output[1] - 7.0) > 0.001){
        printf("ERROR:test_create_network:incorrect network response:%lf,%lf \n", output[0],output[1]);
        return 1;
    }
    return 0;
}
int test_create_network_proto(){
    Ffnn__Network proto_network = FFNN__NETWORK__INIT;
    
    Ffnn__Weight proto_weight1 = FFNN__WEIGHT__INIT;
    Ffnn__Weight proto_weight2 = FFNN__WEIGHT__INIT;
    Ffnn__Bias proto_bias1 = FFNN__BIAS__INIT;
    Ffnn__Bias proto_bias2 = FFNN__BIAS__INIT;
    //Ffnn__Weight * proto_weight1 = (Ffnn__Weight *) malloc(sizeof(Ffnn__Weight));
    //ffnn__weight__init(proto_weight1);
    //Ffnn__Weight * proto_weight2 = (Ffnn__Weight *) malloc(sizeof(Ffnn__Weight));
    //ffnn__weight__init(proto_weight2);
    //Ffnn__Bias * proto_bias1 = (Ffnn__Bias *) malloc(sizeof(Ffnn__Bias));
    //ffnn__bias__init(proto_bias1);
    //Ffnn__Bias * proto_bias2 = (Ffnn__Bias *) malloc(sizeof(Ffnn__Bias));
    //ffnn__bias__init(proto_bias2);

    // Set layers
    int32_t* layer_sizes =(int32_t*) malloc(3 * sizeof(int32_t));
    layer_sizes[0] = 3; layer_sizes[1] = 2; layer_sizes[2] = 2;
    proto_network.layersizes = layer_sizes;
    proto_network.n_layersizes = 3;
    
    // Set activations
    Ffnn__Network__ActivationType* activations = (Ffnn__Network__ActivationType*) malloc(2* sizeof(int32_t));
    activations[0] = FFNN__NETWORK__ACTIVATION_TYPE__RELU;
    activations[1] = FFNN__NETWORK__ACTIVATION_TYPE__SOFTMAX;
    proto_network.activations = activations;
    proto_network.n_activations = 2;

    // Set weights
    proto_weight1.col=3;proto_weight1.row=2;proto_weight1.n_grid = 6;
    double * weight1 = (double*) malloc(6* sizeof(double));
    weight1[0] = 0.0;weight1[1] = 1.0;weight1[2] = 2.0;weight1[3] = 3.0;weight1[4] = 4.0;weight1[5] = 5.0;
    proto_weight1.grid = weight1;

    proto_weight2.col=2;proto_weight2.row=2;proto_weight2.n_grid = 4;
    double * weight2 = (double*) malloc(4* sizeof(double));
    weight2[0] = 0.0;weight2[1] = 1.0;weight2[2] = 2.0;weight2[3] = 3.0;
    proto_weight2.grid = weight2;

    Ffnn__Weight** weights = (Ffnn__Weight**) malloc(2 * sizeof(Ffnn__Weight*));
    weights[0] = &proto_weight1; weights[1] = &proto_weight2;
    proto_network.weights = weights;
    proto_network.n_weights = 2;

    // Set biases
    proto_bias1.n_vector = 2;
    double * bias1 = (double*) malloc(2* sizeof(double));
    bias1[0] = 0.0;bias1[1] = 1.0;
    proto_bias1.vector = bias1;

    proto_bias2.n_vector = 2;
    double * bias2 = (double*) malloc(2* sizeof(double));
    bias2[0] = 0.0;bias2[1] = 1.0;
    proto_bias2.vector = bias2;

    Ffnn__Bias** biases = (Ffnn__Bias**) malloc(2 * sizeof(Ffnn__Bias*));
    biases[0] = &proto_bias1; biases[1] = &proto_bias2;
    proto_network.biases = biases;
    proto_network.n_biases = 2;

    size_t message_length = ffnn__network__get_packed_size(&proto_network);
    uint8_t * protobuf = (uint8_t*)malloc(message_length);
    ffnn__network__pack(&proto_network, protobuf);

    // free useless variables
    free(weight1); free(weight2); free(bias1); free(bias2);
    free(layer_sizes); free(activations);free(biases); free(weights); 

    Network * network = create_network_from_protobuf(protobuf, message_length);
    if(network == NULL){
        printf("ERROR:test_create_network_proto:unable to parse network.\n");
        return 1;
    }
    free(protobuf);
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
    if(test_create_network_proto() == 0){
        printf("*** SUCCESS:test_create_network_proto.\n");
    }else{
        printf("*** FAILURE:test_create_network_proto!\n");
        return 1;
    }

    printf("========= SUCCESS:No failure detected. =========\n");
    return 0;
}


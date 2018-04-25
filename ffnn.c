
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
    printf("Network:create_network:\n %s \n", json_network);
    
    jsmn_parser p;
	jsmntok_t tokens[MAXIMUM_JSON_TOKEN_SIZE];

	jsmn_init(&p);
	int element_count = jsmn_parse(&p, json_network, strlen(json_network), tokens, MAXIMUM_JSON_TOKEN_SIZE);//sizeof(tokens)/sizeof(tokens[0]));
	if (element_count < 0) {
		printf("Network:create_network:Failed to parse JSON: %d\n", element_count);
		return NULL;
	}

	/* Assume the top-level element is an object */
	if (element_count < 1 || tokens[0].type != JSMN_OBJECT) {
		printf("Network:create_network:JSON object expected\n");
		return NULL;
	}

    Network* network = (Network *) malloc(sizeof(Network));

    printf("Network:create_network:parameters------------:\n");
    // Declare temporary parameters for constructing network layers
    char * activation_universal = NULL;
    char ** activations = NULL;// Use alloca or malloc+free
    int number_of_layers = 0;
    int activation_size = 0;
    double ** layer_biases = NULL;// Use alloca or malloc+free
    int layer_biases_size = 0;
    double ** layer_weights = NULL;// Use alloca or malloc+free
    int * layer_weight_cols = NULL;// Use alloca or malloc+free
    int * layer_weight_rows = NULL;// Use alloca or malloc+free
    int layer_weights_size = 0;

    int token_index = 1;
	/* Loop over all keys of the root object */
    while (token_index < element_count) {
        if (json_key_check(json_network, &tokens[token_index], "activations") == 0) {
            jsmntok_t *activation_values = &tokens[++token_index];
			if (activation_values->type != JSMN_ARRAY) {
                printf("ERROR:Network:create_network:Invalid activation format:activations is not an array!");
                break; /// We expect groups to be an array of strings 
            }
            printf("-- activations:\n");
            ++ token_index; // Unwrap array.
            activations = (char**) malloc(activation_values -> size * sizeof(char*));
            activation_size = activation_values -> size;
			for (int i = 0; i < activation_values -> size; i++) {
				jsmntok_t *value_token = &tokens[token_index+i];
                activations[i] = strndup(json_network + value_token->start, value_token->end - value_token->start);
				printf("---- %s\n", activations[i]);
			}
			token_index += activation_values -> size;
		} else if (json_key_check(json_network, &tokens[token_index], "activation") == 0) {
			/// We may use strndup() to fetch string value
            activation_universal = strndup(json_network + tokens[token_index + 1].start, tokens[token_index + 1].end-tokens[token_index + 1].start);
            printf("-- universal activation: %s\n", activation_universal);
            token_index += 2;
		} else if(json_key_check(json_network, &tokens[token_index], "layerSizes") == 0){
            jsmntok_t *json_layer_sizes = &tokens[++token_index];
			if (json_layer_sizes->type != JSMN_ARRAY) {
                printf("ERROR:Network:create_network:Invalid network format:layerSizes is not an array!");
                break; /// We expect groups to be an array of strings 
            }
            network -> layer_sizes = (int *) alloca(json_layer_sizes -> size * sizeof(int));
            number_of_layers = json_layer_sizes -> size - 1;
            printf("-- numberOfLayers:%i (Not including input layer)\n", number_of_layers);
            ++ token_index; // Unwrap array.
            printf("-- layerSizes:\n");
			for (int i = 0; i < json_layer_sizes -> size; i++) {
				jsmntok_t *value_token = &tokens[token_index+i];
                char* layer_size = strndup(json_network + value_token->start, value_token->end - value_token->start);
				printf("---- %s\n", layer_size);
                network -> layer_sizes[i] = atoi(layer_size);
                if(network -> layer_sizes[i] == 0) {
                    printf("ERROR:Network:create_network:Invalid node size in layerSizes is not an integer: %s!", layer_size);
                    free(network);
                    return NULL;
                }
			}
			token_index += json_layer_sizes -> size;
        }
        else if(json_key_check(json_network, &tokens[token_index], "biases") == 0){
            jsmntok_t *bias_objects = &tokens[++token_index];
            if (bias_objects->type != JSMN_ARRAY) {
                printf("ERROR:Network:create_network:Invalid network format:biases is not an array!");
                break; /// We expect groups to be an array of objects 
            }
            ++ token_index; // Unwrap array.
            printf("-- biases:\n");
            layer_biases = (double **) alloca(bias_objects -> size * sizeof(double *));
            for(int bias_ind = 0; bias_ind < bias_objects -> size; bias_ind ++){
                jsmntok_t *bias_object = &tokens[token_index];
                ++ token_index; // Unwrap biasObject

                for(int bias_object_token = 0; bias_object_token < bias_object -> size; bias_object_token++){
                    if(json_key_check(json_network, &tokens[token_index + 1], "vector")){
                        ++ token_index; // access vector value
                        jsmntok_t *bias_object_vector = &tokens[token_index];
                        layer_biases[bias_ind] = (double *) alloca(bias_object_vector -> size * sizeof(double));
                        printf("---- vector:\n");
                        for (int i = 0; i < bias_object_vector -> size; i++) {
                            jsmntok_t *value = &tokens[token_index+i + 1];
                            char* bias_value_str = strndup(json_network + value->start, value->end - value->start);
                            layer_biases[bias_ind][i] = atof(bias_value_str);
                            printf("---- %lf\n", layer_biases[bias_ind][i]);
                        }
                        token_index += bias_object_vector -> size;
                    }
                }
                ++ token_index;// Go to next object
            }
        } else {
            printf("Unexpected key: %.*s\n", tokens[token_index].end-tokens[token_index].start, json_network + tokens[token_index].start);
            ++ token_index;
        }
        /*
        else if(json_key_check(jsonNetwork, &tokens[tokenIndex], "weights") == 0){
            jsmntok_t * weightObjects = &tokens[++tokenIndex];
            if (weightObjects->type != JSMN_ARRAY) {
                printf("ERROR:Invalid network format:weights is not an array!");
                break; /// We expect groups to be an array of strings 
            }
            printf("* weights:\n");
            
            for(int weightIndex = 0; weightIndex < weightObjects -> size; weightIndex ++){
                jsmntok_t *weightObject = &tokens[tokenIndex + 1];
                //printf("DEBUG: weight objects: %.*s\n", weightObject->end - weightObject->start, jsonNetwork + weightObject->start);
                //printf("DEBUG: weight size: %d\n", weightObject->size);
                ++ tokenIndex; // Unwrap weightObject
                for(int weightObjectToken = 0; weightObjectToken < weightObject -> size; weightObjectToken++){
                    if(json_key_check(jsonNetwork, &tokens[tokenIndex + 1], "col")){
                        jsmntok_t *colToken = &tokens[tokenIndex + 2];
                        printf("*-- col: %.*s\n", colToken->end - colToken->start, jsonNetwork + colToken->start);
                        tokenIndex +=2;
                    }else if(json_key_check(jsonNetwork, &tokens[tokenIndex + 1], "row")){
                        jsmntok_t *colToken = &tokens[tokenIndex + 2];
                        printf("*-- row: %.*s\n", colToken->end - colToken->start, jsonNetwork + colToken->start);
                        tokenIndex +=2;
                    }else if(json_key_check(jsonNetwork, &tokens[tokenIndex + 1], "grid")){
                        tokenIndex +=2; // access vector's value & unwrap array
                        jsmntok_t *weightObjectGrid = &tokens[tokenIndex];
                        printf("*-- grid:\n");

                        for (int i = 0; i < weightObjectGrid -> size; i++) {
                            jsmntok_t *value = &tokens[tokenIndex+i + 1];
                            printf("*---- %.*s\n", value->end - value->start, jsonNetwork + value->start);
                        }
                        tokenIndex += weightObjectGrid -> size;
                    }else{
                        printf("ERROR:Invalid network format:Unexpected key in weight object! %.*s", tokens[tokenIndex+1].end-tokens[tokenIndex+1].start, jsonNetwork + tokens[tokenIndex+1].start);
                    }
                }
            }
            ++ tokenIndex;
        }
        */
    }
    printf("DEBUG:Memory activation: %s\n", activations[0]);
    printf("DEBUG:Memory bias: %lf\n", layer_biases[0][1]);
    
    // Validate layer variables
    if(activation_size > 0 && activation_size != network -> number_of_layers){//activation_size > 0 && 
        printf("ERROR:Network:create_network:Activation array specified but the size is not the same as number of layers!\n");
        free_network(network);
        return NULL;
    }
    
    // Free up memory
    //if(activations != NULL)free(activations);
    //if(layer_biases != NULL) free(layer_biases);
    //free(layer_weights);
    //free(layer_weight_cols);
    //if(layer_weight_rows != NULL)free(layer_weight_rows);
    return network;
}

void free_network(Network* network){
    printf("DEBUG:test %i\n",network -> number_of_layers);
    if(network -> number_of_layers > 0){
        printf("DEBUG:layer initialized %i\n",network -> number_of_layers);
        for(int i = 0; i < network -> number_of_layers; i ++){
            free_layer(& (network -> layers[i]));
        }
    }
    free(network);
}

double * run (Network* network, double * input){
    return 0;
}


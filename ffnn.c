
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
		printf("Object expected\n");
		return NULL;
	}

    Network* network = (Network *) malloc(sizeof(Network));

    printf("Network:create_network:parameters:\n");
    int token_index = 1;
    //char * activationHidden = ;
	/* Loop over all keys of the root object */
    while (token_index < element_count) {
        if (json_key_check(json_network, &tokens[token_index], "activationHidden") == 0) {
			/// We may use strndup() to fetch string value
            network -> activation_hidden = strndup(json_network + tokens[token_index + 1].start, tokens[token_index + 1].end-tokens[token_index + 1].start);
			printf("*Network:create_network:parameters-activationHidden: %s\n", network -> activation_hidden);
            token_index += 2;
		}if (json_key_check(json_network, &tokens[token_index], "activationOutput") == 0) {
			/// We may use strndup() to fetch string value
            network -> activation_output = strndup(json_network + tokens[token_index + 1].start, tokens[token_index + 1].end-tokens[token_index + 1].start);
			printf("*Network:create_network:parameters-activationOutput: %s\n", network -> activation_output);
            token_index += 2;
		}
        else {
            printf("Unexpected key: %.*s\n", tokens[token_index].end-tokens[token_index].start, json_network + tokens[token_index].start);
            ++ token_index;
        }
        /*
        else if (json_key_check(jsonNetwork, &tokens[tokenIndex], "activationOutput") == 0) {
			/// We may use strndup() to fetch string value
			printf("* ActivationOutput: %.*s\n", tokens[tokenIndex + 1].end-tokens[tokenIndex + 1].start,
					jsonNetwork + tokens[tokenIndex + 1].start);
            tokenIndex += 2;
		}else if(json_key_check(jsonNetwork, &tokens[tokenIndex], "layerSizes") == 0){
            jsmntok_t *layerValues = &tokens[++tokenIndex];
			if (layerValues->type != JSMN_ARRAY) {
                printf("ERROR:Invalid network format:layerSizes is not an array!");
                break; /// We expect groups to be an array of strings 
            }
            printf("* LayerSizes:\n");
            ++ tokenIndex; // Unwrap array.
			for (int i = 0; i < layerValues -> size; i++) {
				jsmntok_t *valueToken = &tokens[tokenIndex+i];
				printf("*-- %.*s\n", valueToken->end - valueToken->start, jsonNetwork + valueToken->start);
			}
			tokenIndex += layerValues -> size;
        } else if(json_key_check(jsonNetwork, &tokens[tokenIndex], "biases") == 0){
            jsmntok_t *biasObjects = &tokens[++tokenIndex];
            if (biasObjects->type != JSMN_ARRAY) {
                printf("ERROR:Invalid network format:biases is not an array!");
                break; /// We expect groups to be an array of strings 
            }
            printf("* biases:\n");
            for(int biasIndex = 0; biasIndex < biasObjects -> size; biasIndex ++){
                jsmntok_t *biasObject = &tokens[tokenIndex + 1];
                ++ tokenIndex; // Unwrap biasObject
                for(int biasObjectToken = 0; biasObjectToken < biasObject -> size; biasObjectToken++){
                    if(json_key_check(jsonNetwork, &tokens[tokenIndex + 1], "vector")){
                        tokenIndex +=2; // access vector's value & unwrap array
                        jsmntok_t *biasObjectVector = &tokens[tokenIndex];
                        printf("*-- vector:\n");

                        for (int i = 0; i < biasObjectVector -> size; i++) {
                            jsmntok_t *value = &tokens[tokenIndex+i + 1];
                            printf("*---- %.*s\n", value->end - value->start, jsonNetwork + value->start);
                        }
                        tokenIndex += biasObjectVector -> size;
                    }
                }
            }
            ++ tokenIndex;
        } else if(json_key_check(jsonNetwork, &tokens[tokenIndex], "weights") == 0){
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
    return network;
}

double * run (Network* network, double * input){
    return 0;
}


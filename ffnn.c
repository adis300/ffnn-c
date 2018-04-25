
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

// double sigmoid_lookup[LOOKUP_SIZE];

// Activation functions
double inline ffnn_activation_linear(double x) { return x;}

double inline ffnn_activation_threshold(double x) {return x > 0;}

double inline ffnn_activation_relu(double x) {
    if(x > 0) return x;
    return 0;
}

double inline ffnn_activation_sigmoid(double x) {
    if (x < -SIGMOID_CUTOFF) return 0.0;
    if (x > SIGMOID_CUTOFF) return 1.0;
    return 1.0 / (1.0 + exp(-x));
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
    if(network_layer){
        free(network_layer -> weights);
        free(network_layer -> biases);
        free(network_layer -> output);
    }
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
    printf("Network:create_network:\n %s \n\n", json_network);
    
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

    Network* network = (Network *) calloc(1,sizeof(Network));
    // network -> number_of_layers = 0;

    printf("Network:create_network:parameters------------:\n");
    // Declare temporary parameters for constructing network layers
    char * activation_universal = NULL;
    char ** activations = NULL;// Use alloca or malloc+free
    int number_of_layers = 0;
    int activation_size = 0;
    double ** layer_biases = NULL;// Use alloca or malloc+free
    int * layer_biases_vector_sizes = NULL;// Use alloca or malloc+free
    int layer_biases_size = 0;
    double ** layer_weights = NULL;// Use alloca or malloc+free
    int * layer_weights_cols = NULL;// Use alloca or malloc+free
    int * layer_weights_rows = NULL;// Use alloca or malloc+free
    int * layer_weights_grid_sizes = NULL;// Use alloca or malloc+free
    int layer_weights_size = 0;

    int token_index = 1;
	/* Loop over all keys of the root object */
    while (token_index < element_count) {
        if (json_key_check(json_network, &tokens[token_index], "activations") == 0) {
            jsmntok_t *activation_values = &tokens[++token_index];
			if (activation_values->type != JSMN_ARRAY) {
                printf("ERROR:Network:create_network:Invalid activation format:activations is not an array!");
                free(network);
                return NULL;
            }
            printf("-- activations:\n");
            ++ token_index; // Unwrap array.
            activations = (char**) alloca(activation_values -> size * sizeof(char*));
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
                free(network);
                return NULL;
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
        } else if(json_key_check(json_network, &tokens[token_index], "biases") == 0){
            jsmntok_t *bias_objects = &tokens[++token_index];
            if (bias_objects->type != JSMN_ARRAY) {
                printf("ERROR:Network:create_network:Invalid network format:biases is not an array!");
                free(network);
                return NULL;
            }
            ++ token_index; // Unwrap array.
            printf("-- biases:\n");
            layer_biases_size = bias_objects -> size;
            layer_biases = (double **) alloca(bias_objects -> size * sizeof(double *));
            layer_biases_vector_sizes = (int *) alloca(bias_objects -> size * sizeof(int));

            for(int bias_ind = 0; bias_ind < bias_objects -> size; bias_ind ++){
                jsmntok_t *bias_object = &tokens[token_index];
                ++ token_index; // Unwrap biasObject

                for(int bias_object_token = 0; bias_object_token < bias_object -> size; bias_object_token++){
                    if(json_key_check(json_network, &tokens[token_index + 1], "vector")){
                        ++ token_index; // access vector value
                        jsmntok_t *bias_object_vector = &tokens[token_index];
                        layer_biases[bias_ind] = (double *) alloca(bias_object_vector -> size * sizeof(double));
                        layer_biases_vector_sizes[bias_ind] = bias_object_vector -> size;
                        printf("---- vector:\n");
                        for (int i = 0; i < bias_object_vector -> size; i++) {
                            jsmntok_t *value = &tokens[token_index+i + 1];
                            char* bias_value_str = strndup(json_network + value->start, value->end - value->start);
                            layer_biases[bias_ind][i] = atof(bias_value_str);
                            printf("------ %lf\n", layer_biases[bias_ind][i]);
                        }
                        token_index += bias_object_vector -> size;
                    }else{
                        printf("ERROR:Network:create_network:Unexpected key in bias object: %.*s\n", tokens[token_index].end-tokens[token_index].start, json_network + tokens[token_index].start);
                        free(network);
                        return NULL;
                    }
                }
                ++ token_index;// Go to next object
            }
        } else if(json_key_check(json_network, &tokens[token_index], "weights") == 0){
            jsmntok_t *weight_objects = &tokens[++token_index];
            if (weight_objects->type != JSMN_ARRAY) {
                printf("ERROR:Network:create_network:Invalid network format:weights is not an array!");
                free(network);
                return NULL;
            }
            ++ token_index; // Unwrap array.
            printf("-- weights:\n");
            layer_weights_size = weight_objects -> size;
            layer_weights = (double **) alloca(weight_objects -> size * sizeof(double *));
            layer_weights_cols = (int *) alloca(weight_objects -> size * sizeof(int));
            layer_weights_rows= (int *) alloca(weight_objects -> size * sizeof(int));
            layer_weights_grid_sizes= (int *) alloca(weight_objects -> size * sizeof(int));

            for(int weight_ind = 0; weight_ind < weight_objects -> size; weight_ind ++){
                jsmntok_t *weight_object = &tokens[token_index];
                ++ token_index; // Unwrap weightObject

                for(int weight_object_token_ind = 0; weight_object_token_ind < weight_object -> size; weight_object_token_ind++){
                    if (json_key_check(json_network, &tokens[token_index], "col") == 0) {
                        char* col_str = strndup(json_network + tokens[token_index + 1].start, tokens[token_index + 1].end-tokens[token_index + 1].start);
                        layer_weights_cols[weight_ind] = atoi(col_str);
                        printf("---- col: %i\n", layer_weights_cols[weight_ind]);
                        token_index += 2;
                    } else if (json_key_check(json_network, &tokens[token_index], "row") == 0) {
                        char* row_str = strndup(json_network + tokens[token_index + 1].start, tokens[token_index + 1].end-tokens[token_index + 1].start);
                        layer_weights_rows[weight_ind] = atoi(row_str);
                        printf("---- row: %i\n", layer_weights_rows[weight_ind]);
                        token_index += 2;
                    } else if(json_key_check(json_network, &tokens[token_index], "grid") == 0){
                        ++ token_index; // access vector value
                        jsmntok_t *weight_object_grid = &tokens[token_index];
                        layer_weights[weight_ind] = (double *) alloca(weight_object_grid -> size * sizeof(double));
                        printf("---- grid:\n");
                        layer_weights_grid_sizes[weight_ind] = weight_object_grid -> size;
                        for (int i = 0; i < weight_object_grid -> size; i++) {
                            jsmntok_t *value = &tokens[token_index+i + 1];
                            char* weight_value_str = strndup(json_network + value->start, value->end - value->start);
                            layer_weights[weight_ind][i] = atof(weight_value_str);
                            printf("------ %lf\n", layer_weights[weight_ind][i]);
                        }
                        token_index += weight_object_grid -> size;
                    }else{
                        printf("ERROR:Network:create_network:Unexpected key in weight object: %.*s\n", tokens[token_index].end-tokens[token_index].start, json_network + tokens[token_index].start);
                        free(network);
                        return NULL;
                    }
                }
                ++ token_index;// Go to next object
            }
        } else {
            //printf("Unexpected key: %.*s\n", tokens[token_index].end-tokens[token_index].start, json_network + tokens[token_index].start);
            //++ token_index;
            printf("ERROR:Network:create_network:Unexpected key JSON object: %.*s\n", tokens[token_index].end-tokens[token_index].start, json_network + tokens[token_index].start);
            free(network);
            return NULL;
        }
    }
    // printf("DEBUG:Memory activation: %s\n", activations[0]);
    // printf("DEBUG:Memory bias: %lf\n", layer_biases[0][1]);
    // printf("DEBUG:Memory weights: %lf\n", layer_weights[0][1]);

    // Validate network local variables
    if(number_of_layers > 0 && number_of_layers == layer_weights_size && number_of_layers == layer_biases_size){
        if(activation_size == number_of_layers || activation_universal != NULL){
            network -> layers = (NetworkLayer **) calloc(number_of_layers, sizeof(NetworkLayer *));
            network -> number_of_layers = number_of_layers;
            int success = 0; char * layer_activation = NULL;
            for (int i = 0; i < number_of_layers; i ++){
                if(layer_biases_vector_sizes[i] != network -> layer_sizes[i + 1]){
                    printf("ERROR:Network:create_network:Invalid bias vector size:\n");
                    success = 1;
                    break;
                }
                if(layer_weights_cols[i] != network -> layer_sizes[i]){
                    printf("ERROR:Network:create_network:Invalid weight col size:\n");
                    success = 1;
                    break;
                }
                if(layer_weights_rows[i] != network -> layer_sizes[i+1]){
                    printf("ERROR:Network:create_network:Invalid weight col size:\n");
                    success = 1;
                    break;
                }
                if(layer_weights_grid_sizes[i] != network -> layer_sizes[i] * network -> layer_sizes[i + 1]){
                    printf("ERROR:Network:create_network:Invalid weight grid size:\n");
                    success = 1;
                    break;
                }
                // Construct neural network layers
                if(activation_size == number_of_layers) layer_activation = activations[i];
                else layer_activation = activation_universal;
                double * weights = (double*) malloc(layer_weights_grid_sizes[i] * sizeof(double));
                double * biases = (double*) malloc(layer_biases_vector_sizes[i] * sizeof(double));

                /*
                for(int j = 0; j < layer_weights_grid_sizes[i]; j ++) {
                    printf("--- DEBUG: weight %lf\n", layer_weights[i][j]);
                    weights[j] = layer_weights[i][j];
                    printf("--- DEBUG: copied weight %lf\n", weights[j]);
                }

                printf("DEBUG: weights %lf, %lf, %lf, %lf\n",layer_weights[i][0],layer_weights[i][1],layer_weights[i][2],layer_weights[i][3]);
                */
                for(int k = 0; k < layer_biases_vector_sizes[i]; k ++) biases[k] = layer_biases[i][k];
                memcpy(weights, layer_weights[i], layer_weights_grid_sizes[i] * sizeof(double));
                memcpy(biases, layer_biases[i], layer_biases_vector_sizes[i] * sizeof(double));
                NetworkLayer * layer = create_layer(network -> layer_sizes[i+1], network -> layer_sizes[i], weights, biases, layer_activation);
                // printf("DEBUG: copied weights %lf, %lf, %lf, %lf\n",weights[0], weights[1],weights[2],weights[3]);
                network -> layers[i] = layer;
            }
            if(success == 0){
                network -> input_length = network -> layer_sizes[0];
                network -> output_length = network -> layer_sizes[number_of_layers];
                network -> output = network -> layers[number_of_layers - 1] -> output;
                return network;
            }
        }
        else{
            printf("ERROR:Network:create_network:Invalid activation:\n");
        }
    }else{
        printf("ERROR:Network:create_network:Number of layers does not match layer biases and layer weights:\n");
    }
    
    // Failed to create a network, free memory and return
    free_network(network);
    return NULL;
    
    // Free up memory
    //free(activations);
    //free(layer_biases);
    //free(layer_weights);
    //free(layer_weight_cols);
    //free(layer_weight_rows);
}

void free_network(Network* network){
    if(network != NULL && network -> number_of_layers > 0){
        // printf("DEBUG:layer initialized %i\n", network -> number_of_layers);
        for(int i = 0; i < network -> number_of_layers; i ++){
            free_layer(network -> layers[i]);
        }
    }
    free(network);
}

double * run_network (Network* network, double * input){
    // Activate first layer with input
    run_layer(network -> layers[0], input);
    for(int i = 1; i < network -> number_of_layers; i ++){
        run_layer(network -> layers[i], network -> layers[i-1] -> output);
    }
    return network -> output;
}


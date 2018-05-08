# ffnn-c
Feed-forward neural network in C programing language

ffnn-c is a minimal high-performance feed-forward neural networks implementation in C. Suitable for embeded systesms to run pre-trained neural networks. This library is useful when you have a pre-trained neural network structure and you want to use it in an embeded system.

The library supports parsing network specified in JSON.

For any questions please contact adis@brainco.tech.

# Running examples and tests with Makefile
```
make clean && make test && ./test
make clean && make example && ./example
```

# Getting started
* Prepare a json network string content
```
    {
        "layerSizes":[3,2,2],
        "activations":["sigmoid","sigmoid"],
        "weights":[
            {"col": 3, "row": 2, "grid":[1,1, 1.0, 1.0, 1.0, 1.0]},
            {"col\": 2, "row": 2, "grid":[1,1,1,1]}
        ],
        "biases":[
            {"vector" :[ 0.25, 0.25]},
            {"vector" :[ 0.5, 0.5]}]}
        ]
    }
```
 - NOTE: The first layerSizes value is the signal input length, weights, biases and activations should be specified for each layer.

* Create a ffnn_network struct from json network content
```
Network* network = create_network(JSON_NETWORK_ACTIVATION_BY_LAYERS);
```

* Run your input through the network
```
double * input = (double *) alloca(3 * sizeof(double));
input[0] = 1.0; input[1] = 1.0; input[2] = 1.0;
double * output = run_network(network, input);
```
- Note: You are in charge of manage the memory of the input

* Retrieve previous output
```
printf("Network saved response: %lf,%lf \n", network -> output[0], network -> output[1]);
```

* Release memory of the network
free_network(your_network);

# Use protobuf to transfer network

## Build proto-buf code from source on a Mac
1. Prepare environments
```
brew install pkg-config
brew install protobuf
brew install automake
brew install libtool

cd into https://github.com/protobuf-c/protobuf-c
./autogen.sh && ./configure && make && make install
```
2. Compile for C
```
protoc --c_out=. network.proto
```




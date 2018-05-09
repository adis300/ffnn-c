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

# Use Protobuf2 to serialize a neural network
```
from utility.network_pb2 import *

layerSizes = [5, 2, 2]
network = Network()
network.layerSizes.extend(layerSizes)
weights = [Weight(), Weight()]

weights[0].col = layerSizes[0]
weights[0].row = layerSizes[1]
weights[0].grid.extend([0,0,0,0,0,0,0,0,0,0])

weights[1].col = layerSizes[1]
weights[1].row = layerSizes[2]
weights[1].grid.extend([0,0,0,0])

biases = [Bias(), Bias()]

biases[0].vector.extend([0,0,0,0])
biases[1].vector.extend([0,0,0,0])

network.activations.extend([Network.RELU, Network.SOFTMAX])
network.weights.extend(weights)
network.biases.extend(biases)

print(network.SerializeToString())
```
# Use Protobuf3 to serialize a neural network
```
from utility.network_pb3 import *

network = Network()
network.layerSizes.append(3)
network.layerSizes.append(2)
network.layerSizes.append(2)

network.activations.append(Network.ActivationType.RELU)
network.activations.append(Network.ActivationType.LINEAR)

weight1 = Weight()
weight1.col = 3
weight1.row = 2
weight1.grid.append(0.1)
weight1.grid.append(0.2)
weight1.grid.append(0.3)
weight1.grid.append(0.4)
weight1.grid.append(0.5)
weight1.grid.append(0.6)
network.weights.append(weight1)

weight2 = Weight()
weight2.col = 2
weight2.row = 2
weight2.grid.append(0.1)
weight2.grid.append(0.2)
weight2.grid.append(0.3)
weight2.grid.append(0.4)
network.weights.append(weight2)

bias1 = Bias()
bias1.vector.append(0.6)
bias1.vector.append(0.7)
bias2 = bias1

network.biases.append(bias1)
network.biases.append(bias2)

print(network.encode_to_bytes())
```



#include "ffnn.h"
#include <stdio.h>
#include <math.h>

int test_create_layer(){
    NetworkLayer* layer = create_layer(3, 4, 0, 0, "relu");
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
    return 0;
}

int main () //(int argc, char *argv[])
{   
    printf("Start ffnn test.\n");
    if(test_create_layer() != 0) {
        printf("FAILURE:test_create_layer.\n");
        return 1;
    }

    printf("SUCCESS:No failure detected.\n");

    return 0;
}


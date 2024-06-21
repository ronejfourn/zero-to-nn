#include "zm_util.h"
#include "zm_layers.h"

#include <stdio.h>

int main() {
    u32 s1[] = {2, 3, 4};
    f32 d1[2][3][4] = {
        {
            {1,2,3,4}, 
            {6,7,8,9},
            {1,2,3,4}, 
        }, 
        {
            {6,7,8,9},
            {1,2,3,4},
            {6,7,8,9},
        },
    };

    zm_tensor i = zm_tensor_create(3, s1, d1);

    zm_layer l[] = {
        zm_layer_flatten(),
        zm_layer_linear(12, 5),
        zm_layer_ReLU(),
        zm_layer_softmax(1),
    };

    zm_sequential seq = zm_sequential_create(zm_arraylen(l), l);
    zm_tensor o =  zm_sequential_forward(&seq, &i);
    zm_tensor_print(&o);
}

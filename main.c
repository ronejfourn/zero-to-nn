#include "zm_util.h"
#include "zm_loss.h"
#include "zm_layers.h"

#include <stdio.h>

int main() {
    u32 xs_s[] = {4, 2};
    f32 xs_v[4][2] = {
        {0,0},
        {0,1},
        {1,0},
        {1,1},
    };
    zm_tensor xs = zm_tensor_create(zm_arraylen(xs_s), xs_s, xs_v, false);

    u32 ys_s[] = {4, 1};
    f32 ys_v[4][1] = {{0}, {1}, {1}, {0}};
    zm_tensor ys = zm_tensor_create(zm_arraylen(ys_s), ys_s, ys_v, false);

    zm_layer l[] = {
        zm_layer_linear(2, 2),
        zm_layer_ReLU(),
        zm_layer_linear(2, 1),
    };

    zm_sequential s = {l, zm_arraylen(l)};
    
    zm_tensor *ypred = zm_sequential_forward(&s, &xs);

    zm_loss mse = zm_loss_mse();
    zm_tensor *loss = zm_loss_calc(&mse, ypred, &ys);

    zm_tensor_print_data(&ys);
    zm_tensor_print_data(ypred);
    zm_tensor_print_data(loss);

    zm_tensor_print_grad(&ys);
    zm_tensor_print_grad(ypred);
    zm_tensor_print_grad(loss);

    zm_tensor_backward(loss);

    zm_tensor_print_grad(&ys);
    zm_tensor_print_grad(ypred);
    zm_tensor_print_grad(loss);

    return 0;
}

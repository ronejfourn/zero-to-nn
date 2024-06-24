#include "common.h"

ZNN_TENSOR_BACKWARD_FXN(znn_layer_backward_ReLU) {
    znn_tensor *prev = this->prev;
    for (u32 i = 0; i < this->size; i ++)
        prev->grad[i] = (prev->data[i] > 0) * this->grad[i];
}

ZNN_LAYER_INIT_FXN(znn_layer_init_ReLU) {
    this->output = znn_tensor_create_from_shape(
            input->dim, znn_copy(input->shape, input->dim * 4));
    this->output.backward = input->grad ?
        znn_layer_backward_ReLU : NULL;
    znn_tensor_require_grad(&this->output);
    znn_tensor_set_prev(&this->output, input);
}

ZNN_LAYER_FORWARD_FXN(znn_layer_forward_ReLU) {
    znn_tensor *input = this->input;
    for (u32 i = 0; i < input->size; i ++)
        this->output.data[i] = input->data[i] * (input->data[i] > 0);
}

znn_layer _znn_layer_ReLU(char *file, u32 line) {
    znn_trace(file, line);
    znn_layer l = {0};
    l.init = znn_layer_init_ReLU;
    l.forward = znn_layer_forward_ReLU;
    return l;
}

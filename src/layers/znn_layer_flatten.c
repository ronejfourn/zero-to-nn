#include "common.h"

ZNN_TENSOR_BACKWARD_FXN(znn_layer_backward_flatten) {
}

ZNN_LAYER_INIT_FXN(znn_layer_init_flatten) {
    this->output = znn_tensor_create_from_data(
            NULL, input->shape[0], input->step[0]);
    this->output.backward = input->grad ?
        znn_layer_backward_flatten : NULL;
    znn_tensor_set_prev(&this->output, input);
}

ZNN_LAYER_FORWARD_FXN(znn_layer_forward_flatten) {
    this->output.data = this->input->data;
    this->output.grad = this->input->grad;
}

ZNN_LAYER_DESTROY_FXN(znn_layer_destroy_flatten) {
    this->output.data = NULL;
    this->output.grad = NULL;
}

znn_layer _znn_layer_flatten(char *file, u32 line) {
    znn_trace(file, line);
    znn_layer l = {0};
    l.init = znn_layer_init_flatten;
    l.forward = znn_layer_forward_flatten;
    l.destroy = znn_layer_destroy_flatten;
    return l;
}

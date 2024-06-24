#include "common.h"

ZNN_TENSOR_BACKWARD_FXN(znn_layer_backward_flatten) {
    znn_tensor *prev = this->prev;
    memcpy(prev->grad, this->grad, this->size);
}

ZNN_LAYER_INIT_FXN(znn_layer_init_flatten) {
    this->output = znn_tensor_create(input->shape[0], input->step[0]);
    this->output.backward = input->grad ?
        znn_layer_backward_flatten : NULL;
    znn_tensor_set_prev(&this->output, input);
    znn_tensor_require_grad(&this->output);
}

ZNN_LAYER_FORWARD_FXN(znn_layer_forward_flatten) {
    memcpy(this->output.data, this->input->data, this->input->size * 4);
}

znn_layer _znn_layer_flatten(char *file, u32 line) {
    znn_trace(file, line);
    znn_layer l = {0};
    l.init = znn_layer_init_flatten;
    l.forward = znn_layer_forward_flatten;
    return l;
}

#include "common.h"

static ZNN_TENSOR_BACKWARD_FXN(_znn__backward_ReLU) {
    znn_tensor *prev = this->prev;
    for (u32 i = 0; i < this->size; i ++)
        prev->grad[i] = (prev->data[i] > 0) * this->grad[i];
}

static ZNN_LAYER_INIT_FXN(_znn__init_ReLU) {
    this->output = znn_tensor_from_shape(
            input->dim, znn_copy(input->shape, input->dim * 4));
    if (input->grad)
        ZNN_FXN_SET(this->output.backward, _znn__backward_ReLU);
    znn_tensor_require_grad(&this->output);
    znn_tensor_set_prev(&this->output, input);
}

static ZNN_LAYER_FORWARD_FXN(_znn__forward_ReLU) {
    znn_tensor *input = this->input;
    for (u32 i = 0; i < input->size; i ++)
        this->output.data[i] = input->data[i] * (input->data[i] > 0);
}

znn_layer _znn_layer_ReLU(char *file, u32 line) {
    znn_trace(file, line);
    znn_layer l = {0};
    ZNN_FXN_SET(l.init, _znn__init_ReLU);
    ZNN_FXN_SET(l.forward, _znn__forward_ReLU);
    return l;
}

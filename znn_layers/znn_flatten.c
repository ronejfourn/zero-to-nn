#include "../znn_common.h"

static ZNN_TENSOR_BACKWARD_FXN(_znn__backward_flatten) {
}

static ZNN_LAYER_INIT_FXN(_znn__init_flatten) {
    this->output = znn_tensor_from_data(
            NULL, input->shape[0], input->size / input->shape[0]);
    if (input->grad) 
        ZNN_FXN_SET(this->output.backward, _znn__backward_flatten);
    znn_tensor_set_prev(&this->output, input);
}

static ZNN_LAYER_FORWARD_FXN(_znn__forward_flatten) {
    this->output.data = this->input->data;
    this->output.grad = this->input->grad;
}

static ZNN_LAYER_DESTROY_FXN(_znn__destroy_flatten) {
    this->output.data = NULL;
    this->output.grad = NULL;
}

znn_layer _znn_layer_flatten(char *file, u32 line) {
    znn_trace(file, line);
    znn_layer l = {0};
    ZNN_FXN_SET(l.init, _znn__init_flatten);
    ZNN_FXN_SET(l.forward, _znn__forward_flatten);
    ZNN_FXN_SET(l.destroy, _znn__destroy_flatten);
    return l;
}

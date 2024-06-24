#include "common.h"

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_ReLU) {
    zm_tensor *prev = this->prev;
    for (u32 i = 0; i < this->size; i ++)
        prev->grad[i] = (prev->data[i] > 0) * this->grad[i];
}

ZM_LAYER_INIT_FXN(zm_layer_init_ReLU) {
    this->output = zm_tensor_create_from_shape(
            input->dim, zm_copy(input->shape, input->dim * 4));
    this->output.backward = input->grad ? 
        zm_layer_backward_ReLU : NULL;
    zm_tensor_require_grad(&this->output);
    zm_tensor_set_prev(&this->output, input);
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_ReLU) {
    zm_tensor *input = this->input;
    for (u32 i = 0; i < input->size; i ++)
        this->output.data[i] = input->data[i] * (input->data[i] > 0);
}

zm_layer _zm_layer_ReLU(char *file, u32 line) {
    zm_trace(file, line);
    zm_layer l = {0};
    l.init = zm_layer_init_ReLU;
    l.forward = zm_layer_forward_ReLU;
    return l;
}

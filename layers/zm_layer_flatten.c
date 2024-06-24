#include "common.h"

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_flatten) {
    zm_tensor *prev = this->prev;
    memcpy(prev->grad, this->grad, this->size);
}

ZM_LAYER_INIT_FXN(zm_layer_init_flatten) {
    this->output = zm_tensor_create(input->shape[0], input->step[0]);
    this->output.backward = input->grad ? 
        zm_layer_backward_flatten : NULL;
    zm_tensor_set_prev(&this->output, input);
    zm_tensor_require_grad(&this->output);
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_flatten) {
    memcpy(this->output.data, this->input->data, this->input->size * 4);
}

zm_layer _zm_layer_flatten(char *file, u32 line) {
    zm_trace(file, line);
    zm_layer l = {0};
    l.init = zm_layer_init_flatten;
    l.forward = zm_layer_forward_flatten;
    return l;
}

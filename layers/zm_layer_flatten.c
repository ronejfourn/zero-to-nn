#include "common.h"

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_flatten) {
    zm_tensor *prev = this->prev;
    if (!prev->grad) return;

    memcpy(prev->grad, this->grad, this->size);
}


ZM_LAYER_FORWARD_FXN(zm_layer_forward_flatten) {
    if (input != this->input) {
        _update(this, input, zm_layer_backward_flatten,
                zm_tensor_create(input->shape[0], input->step[0]));
        zm_tensor_set_prev(&this->output, input);
    }

    memcpy(this->output.data, input->data, input->size * 4);
}

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_flatten) {
    zm_tensor_destroy(this.output);
}

zm_layer _zm_layer_flatten(char *file, u32 line) {
    zm_trace(file, line);
    L(flatten);
    return l;
}

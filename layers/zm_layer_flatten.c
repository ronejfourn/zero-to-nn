#include "common.h"

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_flatten) {
    zm_tensor *prev = this->prev;
    if (!prev->grad) return;

    memcpy(prev->grad, this->grad, this->_size_flat);
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_flatten) {
    if (input != this->input) {
        this->input = input;
        u32 shape[] = {input->shape[0], input->_offs[0]};
        zm_tensor_destroy(this->output);
        this->output = zm_tensor_create(2, shape, NULL, true);
        zm_tensor_set_prev(&this->output, input, 1);
        this->output.backward = zm_layer_backward_flatten;
    }
    memcpy(this->output.data, input->data, input->_size_flat * sizeof(f32));
}

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_flatten) {
    this.input = NULL;
    zm_tensor_destroy(this.output);
}

zm_layer _zm_layer_flatten(char *file, u32 line) {
    zm_trace(file, line);
    L(flatten);
    return l;
}

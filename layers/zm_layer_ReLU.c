#include "common.h"

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_ReLU) {
    zm_tensor *prev = this->prev;
    if (!prev->grad) return;

    for (u32 i = 0; i < this->_size_flat; i ++)
        prev->grad[i] = (prev->data[i] > 0) * this->grad[i];
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_ReLU) {
    if (input != this->input) {
        this->input = input;
        zm_tensor_destroy(this->output);
        this->output = zm_tensor_create(input->dim, input->shape, NULL, true);
        zm_tensor_set_prev(&this->output, input, 1);
        this->output.backward = zm_layer_backward_ReLU;
    }

    for (u32 i = 0; i < input->_size_flat; i ++)
        this->output.data[i] = input->data[i] * (input->data[i] > 0);
}

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_ReLU) {
    zm_tensor_destroy(this.output);
}

zm_layer _zm_layer_ReLU(char *file, u32 line) {
    zm_trace(file, line);
    L(ReLU);
    return l;
}

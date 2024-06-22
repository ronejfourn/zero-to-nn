#include "common.h"

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_ReLU) {
    zm_tensor *prev = this->prev;
    if (!prev->grad) return;

    for (u32 i = 0; i < this->size; i ++)
        prev->grad[i] = (prev->data[i] > 0) * this->grad[i];
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_ReLU) {
    if (input != this->input) {
        _update(this, input, zm_layer_backward_ReLU,
                zm_tensor_create_from_shape(input->dim, zm_copy(input->shape, input->dim * 4)));
        zm_tensor_set_prev(&this->output, input);
    }

    for (u32 i = 0; i < input->size; i ++)
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

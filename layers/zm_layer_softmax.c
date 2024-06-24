#include "common.h"

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_softmax) {
    uintptr_t dim = (uintptr_t)this->backward_data;
    zm_tensor *prev = this->prev;
    u32 I = dim ? prev->step[dim - 1] : prev->size;
    u32 J = prev->step[dim];
    u32 K = prev->shape[dim] * J;

    for (int i = 0; i < prev->size; i += I) {
        f32 *oi = this->data + i, *gi = prev->grad + i;
        for (u32 j = 0; j < J; j ++) {
            f32 *oj = oi + j, *gj = gi + j;
            for (u32 k = 0; k < K; k += J)
                for (u32 l = 0; l < K; l += J)
                    gj[k] += oj[l] * ((l == k) - oj[k]);
        }
    }
}

ZM_LAYER_INIT_FXN(zm_layer_init_softmax) {
    u32 dim = (u32)(uintptr_t)this->parameters;
    assert(dim < input->dim);
    this->output = zm_tensor_create_from_shape(
            input->dim, zm_copy(input->shape, input->dim * 4));
    this->output.backward = input->grad ? 
        zm_layer_backward_softmax : NULL;
    this->output.backward_data = (void *)(uintptr_t)dim;
    zm_tensor_require_grad(&this->output);
    zm_tensor_set_prev(&this->output, input);
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_softmax) {
    u32 dim = (u32)(uintptr_t)this->parameters;
    zm_tensor *input = this->input;
    zm_tensor *output = &this->output;

    u32 I = dim ? input->step[dim - 1] : input->size;
    u32 J = input->step[dim];
    u32 K = input->shape[dim] * J;

    for (int i = 0; i < input->size; i += I) {
        f32 *oi = output->data + i;
        for (u32 j = 0; j < J; j ++) {
            f32 sum = 0;
            f32 *oj = oi + j;
            for (u32 k = 0; k < K; k += J)
                sum += (oj[k] = exp(input->data[i + j + k]));

            for (u32 k = 0; k < K; k += J)
                oj[k] /= sum;
        }
    }
}

zm_layer _zm_layer_softmax(u32 dim, char *file, u32 line) {
    zm_trace(file, line);

    zm_layer l = {0};
    l.init = zm_layer_init_softmax;
    l.forward = zm_layer_forward_softmax;
    l.parameters = (void*)(uintptr_t)dim;

    return l;
}

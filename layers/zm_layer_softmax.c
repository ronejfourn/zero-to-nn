#include "common.h"

#include <math.h>

typedef struct {
    u32 dim;
} zm_layer_data_softmax;

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_softmax) {
    assert(this->prev);

    zm_tensor *prev = this->prev[0];
    if (!prev->grad) return;

    memset(prev->grad, 0, prev->_size_flat);

    uintptr_t dim = (uintptr_t)this->backward_data;
    u32 I = dim ? prev->_offs[dim - 1] : prev->_size_flat;
    u32 J = prev->_offs[dim];
    u32 K = prev->shape[dim] * J;

    for (int i = 0; i < prev->_size_flat; i += I) {
        f32 *oi = this->data + i;
        f32 *gi = prev->grad + i;
        for (u32 j = 0; j < J; j ++) {
            f32 *oj = oi + j;
            f32 *gj = gi + j;
            for (u32 k = 0; k < K; k += J)
                for (u32 l = 0; l < K; l += J)
                    gj[k] += oj[l] * ((l == k) - oj[k]);
        }
    }
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_softmax) {
    zm_layer_data_softmax *ld = this->layer_data;
    if (input != this->input) {
        this->input = input;
        assert(ld->dim < input->dim);
        zm_tensor_destroy(this->output);
        this->output = zm_tensor_create(input->dim, input->shape, NULL, true);

        zm_tensor *p[] = {input};
        _set_prev(&this->output, p, 1);
        this->output.backward = zm_layer_backward_softmax;
        this->output.backward_data = (void *)(uintptr_t)ld->dim;
    }

    zm_tensor *output = &this->output;
    u32 I = ld->dim ? input->_offs[ld->dim - 1] : input->_size_flat;
    u32 J = input->_offs[ld->dim];
    u32 K = input->shape[ld->dim] * J;

    for (int i = 0; i < input->_size_flat; i += I) {
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

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_softmax) {
    this.input = NULL;
    zm_tensor_destroy(this.output);
    zm_free(this.layer_data);
}

zm_layer _zm_layer_softmax(u32 dim, char *file, u32 line) {
    zm_trace(file, line);
    zm_layer_data_softmax *data = zm_malloc(sizeof(*data));
    data->dim = dim;

    L(softmax);
    l.layer_data = data;

    return l;
}

